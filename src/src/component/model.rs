use std::{fs::File, path::PathBuf};

use candle_core::Device;
use candle_nn::{Embedding, LayerNorm, VarBuilder};
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

pub struct ModelLoader {
    model: ModelWrapper,
    tokenizer: Tokenizer,
}

pub struct ModelWrapper {
    pub buffer: memmap2::Mmap,
    pub model_filename: PathBuf,
    pub device: Device,
    pub config: Config,
}

#[derive(serde::Deserialize, Debug)]
pub struct Config {
    pub _name_or_path: Option<String>,
    pub activation_function: String,
    pub architectures: Vec<String>,
    pub attn_pdrop: f32,
    pub bos_token_id: i64,
    pub embd_pdrop: f32,
    pub eos_token_id: i64,
    pub initializer_range: f32,
    pub layer_norm_epsilon: f64,
    pub model_type: String,
    pub n_embd: usize,
    pub n_head: i64,
    pub n_inner: Option<usize>,
    pub n_layer: i64,
    pub n_positions: i64,
    pub pad_token_id: i64,
    pub quantization_config: Option<QuantizationConfig>,
    pub resid_pdrop: f32,
    pub rotary_dim: i64,
    pub rotary_pct: f32,
    pub scale_attn_weights: bool,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: i64,
}

#[derive(serde::Deserialize, Debug)]
pub struct QuantizationConfig {
    pub bits: i32,
    pub group_size: i32,
    pub quant_method: String,
    pub version: String,
    pub zero_point: bool,
}

pub struct HiddenLayer {
    pub layer_norm: LayerNorm,
    pub attention: Attention,
    pub mlp: MLP,
}

pub struct Attention {
    pub q: Embedding,
    pub k: Embedding,
    pub v: Embedding,
    pub out: Embedding,
}

pub struct MLP {
    pub fc_in: Embedding,
    pub fc_out: Embedding,
}

impl ModelLoader {
    pub fn new(model_dir: &str, tokenizer_dir: &str, device: &Device) -> ModelLoader {
        println!("Begin loading model...");

        let model_dir = std::path::Path::new(model_dir);

        let config_filename = model_dir.join("config.json");
        let config: Config = serde_json::from_reader(File::open(config_filename).unwrap()).unwrap();

        let model_filename = model_dir.join("model.safetensors");
        let buffer = Self::load_model(&model_filename);
        let model = ModelWrapper {
            model_filename,
            buffer,
            device: device.clone(),
            config,
        };
        model.load_weights();

        let tokenizer_dir = std::path::Path::new(tokenizer_dir);
        let tokenizer_filename = tokenizer_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        println!("Loading model done");

        Self { model, tokenizer }
    }

    pub fn load_model(model_filename: &PathBuf) -> Mmap {
        let file = File::open(model_filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };

        buffer
    }

    pub fn get_tensors(&self) -> SafeTensors {
        self.model.tensors()
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn get_config(&self) -> &Config {
        &self.model.config
    }
}

impl ModelWrapper {
    fn tensors(&self) -> SafeTensors {
        let tensors = SafeTensors::deserialize(&self.buffer).unwrap();

        return tensors;
    }

    fn load_weights(&self) {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.model_filename.clone()],
                candle_core::DType::F16,
                &self.device,
            )
            .unwrap()
        };

        let hidden_layers_var = vb.pp("transformer.h");
        let inner_size = self.config.n_inner.unwrap_or(4 * self.config.n_embd);

        let layers = (0..self.config.n_layer)
            .map(|i| {
                let layer_var = hidden_layers_var.pp(&format!("{}", i));
                let layer = HiddenLayer::new(
                    layer_var,
                    self.config.n_embd,
                    inner_size,
                    self.config.layer_norm_epsilon,
                );

                layer
            })
            .collect::<Vec<_>>();

        let transformers = vb.pp("transformer");
        println!(
            "Exists? - {}",
            transformers.contains_tensor("h.15.ln_1.weight")
        );
    }
}

impl HiddenLayer {
    pub fn new(
        layer_var: VarBuilder,
        embed_size: usize,
        inner_size: usize,
        lm_eps: f64,
    ) -> HiddenLayer {
        let layer_norm_var = layer_var.pp("ln_1");
        let attn_var = layer_var.pp("attn");
        let mlp_var = layer_var.pp("mlp");

        let layer_norm = LayerNorm::new(
            layer_norm_var.get(embed_size, "weight").unwrap(),
            layer_norm_var.get(embed_size, "bias").unwrap(),
            lm_eps,
        );

        let attention = Attention::new(attn_var, embed_size);

        let mlp = MLP::new(mlp_var, inner_size, embed_size);

        HiddenLayer {
            layer_norm,
            attention,
            mlp,
        }
    }
}

impl Attention {
    pub fn new(vb: VarBuilder, embed_size: usize) -> Attention {
        let q = Embedding::new(vb.get(embed_size, "q_proj.weight").unwrap(), embed_size);
        let k = Embedding::new(vb.get(embed_size, "k_proj.weight").unwrap(), embed_size);
        let v = Embedding::new(vb.get(embed_size, "v_proj.weight").unwrap(), embed_size);
        let out = Embedding::new(vb.get(embed_size, "out_proj.weight").unwrap(), embed_size);

        Attention { q, k, v, out }
    }
}

impl MLP {
    pub fn new(vb: VarBuilder, inter_size: usize, embed_size: usize) -> MLP {
        let fc_in = Embedding::new(vb.get(embed_size, "fc_in.weight").unwrap(), inter_size);
        let fc_out = Embedding::new(vb.get(inter_size, "fc_out.weight").unwrap(), embed_size);

        MLP { fc_in, fc_out }
    }
}
