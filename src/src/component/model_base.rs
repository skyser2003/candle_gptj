#[derive(serde::Deserialize, Debug, Clone)]
pub struct QuantizationConfig {
    pub bits: i32,
    pub group_size: i32,
    pub quant_method: String,
    pub version: String,
    pub zero_point: bool,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct GPTJConfig {
    pub _name_or_path: Option<String>,
    pub activation_function: String,
    pub architectures: Vec<String>,
    pub attn_pdrop: f32,
    pub bos_token_id: usize,
    pub embd_pdrop: f32,
    pub eos_token_id: usize,
    pub initializer_range: f32,
    pub layer_norm_epsilon: f64,
    pub model_type: String,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub n_layer: usize,
    pub n_positions: usize,
    pub pad_token_id: Option<usize>,
    pub quantization_config: Option<QuantizationConfig>,
    pub resid_pdrop: f32,
    pub rotary_dim: Option<usize>,
    pub rotary_pct: Option<f32>,
    pub scale_attn_weights: bool,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}
