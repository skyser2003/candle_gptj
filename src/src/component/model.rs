use std::{fs::File, path::PathBuf};

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

pub struct ModelLoader {
    model: ModelWrapper,
    tokenizer: Tokenizer,
    config: Config,
}

pub struct ModelWrapper {
    pub buffer: memmap2::Mmap,
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
    pub layer_norm_epsilon: f32,
    pub model_type: String,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_inner: Option<i64>,
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

impl ModelLoader {
    pub fn new(model_dir: &str, tokenizer_dir: &str) -> ModelLoader {
        println!("Begin loading model...");

        let model_dir = std::path::Path::new(model_dir);
        let model_filename = model_dir.join("model.safetensors");
        let model = Self::load_model(&model_filename);

        let tokenizer_dir = std::path::Path::new(tokenizer_dir);
        let tokenizer_filename = tokenizer_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        let config_filename = model_dir.join("config.json");
        let config: Config = serde_json::from_reader(File::open(config_filename).unwrap()).unwrap();

        println!("Loading model done");

        Self {
            model,
            tokenizer,
            config,
        }
    }

    pub fn load_model(model_filename: &PathBuf) -> ModelWrapper {
        let file = File::open(model_filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };

        let wrapper = ModelWrapper { buffer: buffer };

        return wrapper;
    }

    pub fn get_tensors(&self) -> SafeTensors {
        self.model.tensors()
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn get_config(&self) -> &Config {
        &self.config
    }
}

impl ModelWrapper {
    pub fn tensors(&self) -> SafeTensors {
        let tensors = SafeTensors::deserialize(&self.buffer).unwrap();

        return tensors;
    }
}
