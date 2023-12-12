use std::{fs::File, path::PathBuf};

use candle_core::Device;
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
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
    pub model: CoreModel,
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
    pub pad_token_id: Option<i64>,
    pub quantization_config: Option<QuantizationConfig>,
    pub resid_pdrop: f32,
    pub rotary_dim: i64,
    pub rotary_pct: Option<f32>,
    pub scale_attn_weights: bool,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(serde::Deserialize, Debug)]
pub struct QuantizationConfig {
    pub bits: i32,
    pub group_size: i32,
    pub quant_method: String,
    pub version: String,
    pub zero_point: bool,
}

pub struct CoreModel {
    pub word_token_embedding: Embedding,
    pub hidden_layers: Vec<HiddenLayer>,
    pub layernorm_final: LayerNorm,
}

pub struct HiddenLayer {
    pub layer_norm: LayerNorm,
    pub attention: Attention,
    pub mlp: MLP,
}

pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
}

pub struct MLP {
    pub fc_in: Linear,
    pub fc_out: Linear,
}

impl ModelLoader {
    pub fn new(model_dir: &str, tokenizer_dir: &str, device: &Device) -> ModelLoader {
        println!("Begin loading model...");

        let model_dir = std::path::Path::new(model_dir);

        let config_filename = model_dir.join("config.json");
        let config: Config = serde_json::from_reader(File::open(config_filename).unwrap()).unwrap();

        let model_filename = model_dir.join("model.safetensors");
        let buffer = Self::load_model(&model_filename);
        let core_model = CoreModel::new(&model_filename, &device, &config);

        let model = ModelWrapper {
            model_filename,
            buffer,
            device: device.clone(),
            config,
            model: core_model,
        };

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
}

impl CoreModel {
    fn new(model_filename: &PathBuf, device: &Device, config: &Config) -> CoreModel {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_filename.clone()],
                candle_core::DType::F16,
                &device,
            )
            .unwrap()
        };

        let tf_vb = vb.pp("transformer");

        let word_token_embedding = Embedding::new(
            tf_vb
                .get((config.vocab_size, config.n_embd), "wte.weight")
                .unwrap(),
            config.n_embd,
        );

        let hidden_layers_vb = tf_vb.pp("h");
        let inner_size = config.n_inner.unwrap_or(4 * config.n_embd);

        let layers = (0..config.n_layer)
            .map(|i| {
                let layer_vb = hidden_layers_vb.pp(&format!("{}", i));
                let layer = HiddenLayer::new(
                    layer_vb,
                    config.n_embd,
                    inner_size,
                    config.layer_norm_epsilon,
                );

                layer
            })
            .collect::<Vec<_>>();

        let layernorm_final = LayerNorm::new(
            tf_vb.get(config.n_embd, "ln_f.weight").unwrap(),
            tf_vb.get(config.n_embd, "ln_f.bias").unwrap(),
            config.layer_norm_epsilon,
        );

        Self {
            word_token_embedding,
            hidden_layers: layers,
            layernorm_final,
        }
    }
}

impl HiddenLayer {
    pub fn new(
        layer_vb: VarBuilder,
        embed_size: usize,
        inner_size: usize,
        lm_eps: f64,
    ) -> HiddenLayer {
        let layer_norm_vb = layer_vb.pp("ln_1");
        let attn_vb = layer_vb.pp("attn");
        let mlp_vb = layer_vb.pp("mlp");

        let layer_norm = LayerNorm::new(
            layer_norm_vb.get(embed_size, "weight").unwrap(),
            layer_norm_vb.get(embed_size, "bias").unwrap(),
            lm_eps,
        );

        let attention = Attention::new(attn_vb, embed_size);

        let mlp = MLP::new(mlp_vb, inner_size, embed_size);

        HiddenLayer {
            layer_norm,
            attention,
            mlp,
        }
    }
}

impl Attention {
    pub fn new(vb: VarBuilder, embed_size: usize) -> Attention {
        let q = Linear::new(
            vb.get((embed_size, embed_size), "q_proj.weight").unwrap(),
            None,
        );
        let k = Linear::new(
            vb.get((embed_size, embed_size), "k_proj.weight").unwrap(),
            None,
        );
        let v = Linear::new(
            vb.get((embed_size, embed_size), "v_proj.weight").unwrap(),
            None,
        );
        let out = Linear::new(
            vb.get((embed_size, embed_size), "out_proj.weight").unwrap(),
            None,
        );

        Attention { q, k, v, out }
    }
}

impl MLP {
    pub fn new(vb: VarBuilder, inter_size: usize, embed_size: usize) -> MLP {
        let fc_in = Linear::new(
            vb.get((inter_size, embed_size), "fc_in.weight").unwrap(),
            Some(vb.get(inter_size, "fc_in.bias").unwrap()),
        );
        let fc_out = Linear::new(
            vb.get((embed_size, inter_size), "fc_out.weight").unwrap(),
            Some(vb.get(embed_size, "fc_out.bias").unwrap()),
        );

        MLP { fc_in, fc_out }
    }
}
