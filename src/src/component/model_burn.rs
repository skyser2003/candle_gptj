use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

use anyhow::Result;
use burn::{
    nn::{Dropout, Embedding, LayerNorm, Linear},
    tensor::{backend::Backend, Device, Tensor},
};
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use super::model_base::GPTJConfig;

pub struct ModelLoader<B: Backend> {
    model: CausalModel<B>,
    tokenizer: Tokenizer,
}

pub struct CausalModel<B: Backend> {
    pub buffer: memmap2::Mmap,
    pub model_filename: PathBuf,
    pub device: Device<B>,
    pub config: GPTJConfig,
    pub transformer: CoreModel<B>,
    pub lm_head: Linear<B>,
}

pub struct CoreModel<B: Backend> {
    pub config: GPTJConfig,
    // pub dtype: DType,
    pub dtype_min: f32,
    pub word_token_embedding: Embedding<B>,
    pub hidden_layers: Vec<HiddenLayer<B>>,
    pub layernorm_final: LayerNorm<B>,
    pub drop: Dropout,
    pub is_parallel: bool,
}

pub struct HiddenLayer<B: Backend> {
    pub layer_norm: LayerNorm<B>,
    pub attention: Attention<B>,
    pub mlp: MLP<B>,
}

pub struct Attention<B: Backend> {
    pub qkv: Linear<B>,
    pub out: Linear<B>,
    pub num_heads: usize,
    pub head_size: usize,
    pub rotary_dim: usize,
    pub embed_size: usize,
    pub scale_attn: Tensor<B, 1>,
    pub bias: Tensor<B, 1>,
    pub embed_positions: Tensor<B, 2>,
    pub attn_dropout: Dropout,
    pub resid_dropout: Dropout,
}

pub struct MLP<B: Backend> {
    pub fc_in: Linear<B>,
    pub fc_out: Linear<B>,
    // pub activation: Activation,
    pub dropout: Dropout,
}

pub struct CausalOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub past_key_values: Vec<(Tensor<B, 2>, Tensor<B, 2>)>,
    pub hidden_states: Vec<Tensor<B, 2>>,
    pub attentions: Vec<Tensor<B, 2>>,
}

impl<B> ModelLoader<B>
where
    B: Backend,
{
    pub fn new(model_dir: &str, tokenizer_dir: &str, device: &Device<B>) -> ModelLoader<B> {
        println!("Begin loading model...");
        let start_time = Instant::now();

        let model_dir = std::path::Path::new(model_dir);

        let config_filename = model_dir.join("config.json");
        let config: GPTJConfig =
            serde_json::from_reader(File::open(config_filename).unwrap()).unwrap();

        let model_filename = model_dir.join("model.safetensors");
        let buffer = Self::load_model(&model_filename);

        let dtype_map = vec![
            ("float16", (DType::F16, -65504.0)),
            ("float32", (DType::F32, f32::MIN)),
            ("bfloat16", (DType::BF16, -3.38e38)),
        ]
        .into_iter()
        .collect::<HashMap<_, _>>();

        let torch_dtype = config.torch_dtype.clone().unwrap_or("".to_string());

        let torch_dtype = if dtype_map.contains_key(torch_dtype.as_str()) {
            torch_dtype
        } else {
            "float32".to_string()
        };

        let (dtype, dtype_min) = *dtype_map.get(torch_dtype.as_str()).unwrap();

        let vb = unsafe { SafeTensors::deserialize(&buffer).unwrap() };

        let core_model = CoreModel::new(&vb, dtype, dtype_min, &device, &config);
        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head")).unwrap();

        let model = CausalModel {
            model_filename,
            buffer,
            device: device.clone(),
            config,
            transformer: core_model,
            lm_head,
        };

        let tokenizer_dir = std::path::Path::new(tokenizer_dir);
        let tokenizer_filename = tokenizer_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        let mut instance = Self { model, tokenizer };

        let _ = instance.inference(&["hot loading"]);

        let end_time = Instant::now();

        println!(
            "Loading model done, {}s",
            (end_time - start_time).as_secs_f32()
        );
        println!("");

        instance
    }

    pub fn load_model(model_filename: &PathBuf) -> Mmap {
        let file = File::open(model_filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };

        buffer
    }

    pub fn inference(&mut self, inputs: &[&str]) -> Result<Vec<String>> {
        Ok(vec![])
    }
}

impl<B> CoreModel<B>
where
    B: Backend,
{
    fn new(
        vb: &SafeTensors,
        dtype: DType,
        dtype_min: f32,
        device: &B::Device,
        config: &GPTJConfig,
    ) -> CoreModel {
        Self {}
    }
}
