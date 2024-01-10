use std::{collections::HashMap, fs::File, path::PathBuf};

use candle_core::IndexOp;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation};
use candle_nn::{ops::softmax, Dropout, Embedding, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
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
    pub lm_head: Linear,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
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

#[derive(serde::Deserialize, Debug, Clone)]
pub struct QuantizationConfig {
    pub bits: i32,
    pub group_size: i32,
    pub quant_method: String,
    pub version: String,
    pub zero_point: bool,
}

pub struct CoreModel {
    pub config: Config,
    pub dtype: DType,
    pub dtype_min: f32,
    pub word_token_embedding: Embedding,
    pub hidden_layers: Vec<HiddenLayer>,
    pub layernorm_final: LayerNorm,
    pub drop: Dropout,
    pub is_parallel: bool,
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
    pub num_heads: usize,
    pub head_size: usize,
    pub rotary_dim: usize,
    pub bias: Tensor,
    pub embed_positions: Tensor,
    pub attn_dropout: Dropout,
    pub resid_dropout: Dropout,
}

pub struct MLP {
    pub fc_in: Linear,
    pub fc_out: Linear,
    pub activation: Activation,
    pub dropout: Dropout,
}

impl ModelLoader {
    pub fn new(model_dir: &str, tokenizer_dir: &str, device: &Device) -> ModelLoader {
        println!("Begin loading model...");

        let model_dir = std::path::Path::new(model_dir);

        let config_filename = model_dir.join("config.json");
        let config: Config = serde_json::from_reader(File::open(config_filename).unwrap()).unwrap();

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

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_filename.clone()], dtype, &device).unwrap()
        };

        let core_model = CoreModel::new(&vb, dtype, dtype_min, &device, &config);
        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head")).unwrap();

        let model = ModelWrapper {
            model_filename,
            buffer,
            device: device.clone(),
            config,
            model: core_model,
            lm_head,
        };

        let tokenizer_dir = std::path::Path::new(tokenizer_dir);
        let tokenizer_filename = tokenizer_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

        println!("Loading model done");
        println!("");

        Self { model, tokenizer }
    }

    pub fn load_model(model_filename: &PathBuf) -> Mmap {
        let file = File::open(model_filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };

        buffer
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.model.model.forward(
            Some(&input_ids),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(false),
            false,
            false,
            false,
            &self.model.device,
        )?;

        let lm_logits = self
            .model
            .lm_head
            .forward(&hidden_states)?
            .to_dtype(DType::F32);

        lm_logits
    }

    pub fn inference(&mut self, inputs: &[&str]) -> Result<Vec<String>> {
        let batch_size = inputs.len();
        let encodings = self.tokenizer.encode_batch(inputs.to_vec(), true).unwrap();
        let tokens = encodings
            .iter()
            .map(|enc| enc.get_ids())
            .collect::<Vec<_>>();

        let input_ids = Tensor::new(tokens, &self.model.device)?;

        let lm_logits = self.forward(&input_ids)?;

        let logits = lm_logits.argmax(D::Minus1)?;
        let logits = logits.to_vec2::<u32>()?;

        let logits = logits
            .iter()
            .map(|nested_vec| nested_vec.as_slice())
            .collect::<Vec<_>>();

        let outputs = self.tokenizer.decode_batch(&logits, true).unwrap();
        return Ok(outputs);

        // TODO top_k, etc
        let next_logits = lm_logits.narrow(1, lm_logits.dim(1)? - 2, 1)?.squeeze(1)?;

        let mut output_ids = vec![];

        for _ in 0..batch_size {
            // TODO LogitsProcessor saved per batch, not every loop
            let mut logits_proc = LogitsProcessor::new(0, None, Some(0.3));
            let next_logit = next_logits.i(0)?;

            let gen_id = logits_proc.sample(&next_logit)?;

            output_ids.push([gen_id]);
        }

        let output_ids = output_ids
            .iter()
            .map(|ids| ids.as_slice())
            .collect::<Vec<_>>();

        let outputs = self.tokenizer.decode_batch(&output_ids, true).unwrap();

        Ok(outputs)
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
    fn new(
        vb: &VarBuilder,
        dtype: DType,
        dtype_min: f32,
        device: &Device,
        config: &Config,
    ) -> CoreModel {
        let tf_vb = vb.pp("transformer");

        let word_token_embedding = Embedding::new(
            tf_vb
                .get((config.vocab_size, config.n_embd), "wte.weight")
                .unwrap(),
            config.n_embd,
        );

        let hidden_layers_vb = tf_vb.pp("h");
        let inner_size = config.n_inner.unwrap_or(4 * config.n_embd);
        let rotary_dim = config.rotary_dim.unwrap_or(0);
        let pos_embed_dimension = if rotary_dim == 0 {
            config.n_embd
        } else {
            rotary_dim
        };

        let layers = (0..config.n_layer)
            .map(|i| {
                let layer_vb = hidden_layers_vb.pp(&format!("{}", i));
                let layer = HiddenLayer::new(
                    layer_vb,
                    config.n_embd,
                    inner_size,
                    config.n_head,
                    config.layer_norm_epsilon,
                    config.n_positions,
                    pos_embed_dimension,
                    rotary_dim,
                    config.attn_pdrop,
                    config.resid_pdrop,
                    device,
                );

                layer
            })
            .collect::<Vec<_>>();

        let layernorm_final = LayerNorm::new(
            tf_vb.get(config.n_embd, "ln_f.weight").unwrap(),
            tf_vb.get(config.n_embd, "ln_f.bias").unwrap(),
            config.layer_norm_epsilon,
        );

        let drop = Dropout::new(config.embd_pdrop);

        Self {
            config: config.clone(),
            dtype,
            dtype_min,
            word_token_embedding,
            hidden_layers: layers,
            layernorm_final,
            drop,
            is_parallel: false,
        }
    }

    fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        past_key_values: Option<Vec<Vec<Option<Tensor>>>>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<Tensor>,
        head_mask: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        use_cache: Option<bool>,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        device: &Device,
    ) -> Result<Tensor> {
        let use_cache = use_cache.unwrap_or(self.config.use_cache);

        let (input_shape, batch_size, device) = if input_ids.is_some() && input_embeds.is_some() {
            panic!("You cannot specify both input_ids and inputs_embeds at the same time")
        } else if let Some(input_ids) = input_ids {
            // TODO
            // self.warn_if_padding_and_no_attention_mask(&input_ids, attention_mask);

            let input_shape = input_ids.shape().clone();
            let input_ids = input_ids.reshape(((), input_ids.dim(D::Minus1)?))?;
            let batch_size = input_ids.dim(0)?;
            let device = input_ids.device().clone();

            (input_shape, batch_size, device)
        } else if let Some(input_embeds) = &input_embeds {
            let input_shape = input_embeds.shape().clone();
            let batch_size = input_embeds.dim(0)?;
            let device = input_embeds.device().clone();

            (input_shape, batch_size, device)
        } else {
            panic!("You have to specify either input_ids or inputs_embeds")
        };

        let token_type_ids = if let Some(token_type_ids) = token_type_ids {
            Some(token_type_ids.reshape(((), token_type_ids.dim(D::Minus1)?))?)
        } else {
            None
        };

        let (past_length, past_key_values) = if past_key_values.is_none() {
            let past_length = 0usize;
            let past_key_values: Vec<Vec<Option<Tensor>>> = vec![vec![]; self.config.n_layer];

            (past_length, past_key_values)
        } else {
            let past_key_values = past_key_values.unwrap();
            let past_length = past_key_values[0][0].as_ref().unwrap().dim(D::Minus2)?;

            (past_length, past_key_values)
        };

        let position_ids = if position_ids.is_none() {
            let position_ids = Tensor::arange(
                past_length as i64,
                *input_shape.dims().last().unwrap() as i64 + past_length as i64,
                &device,
            )?;
            position_ids.unsqueeze(0)?
        } else {
            position_ids.unwrap()
        };

        let attention_mask = if let Some(attention_mask) = attention_mask {
            if batch_size <= 0 {
                panic!("Batch size has to be defined and > 0");
            }

            let attention_mask = attention_mask.reshape((batch_size, ()))?;
            let attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?;
            let attention_mask = attention_mask.to_dtype(self.dtype)?;
            let attention_mask = ((1.0 - attention_mask)? * self.dtype_min as f64)?;

            Some(attention_mask)
        } else {
            None
        };

        let head_mask = Self::get_head_mask(head_mask, self.config.n_layer, false, self.dtype);

        let input_embeds = if input_embeds.is_none() {
            self.word_token_embedding.forward(&input_ids.unwrap())?
        } else {
            input_embeds.unwrap()
        };

        let mut hidden_states = input_embeds;

        if let Some(token_type_ids) = token_type_ids {
            let token_type_embeds = self.word_token_embedding.forward(&token_type_ids)?;
            hidden_states = (hidden_states + token_type_embeds)?;
        }

        hidden_states = self.drop.forward(&hidden_states, false)?;

        let mut partial_output_shape = Vec::new();

        for i in input_shape.dims().iter().skip(1) {
            partial_output_shape.push(*i);
        }

        partial_output_shape.push(hidden_states.dim(D::Minus1)?);

        let mut presents = Vec::new();
        let mut all_self_attentions = Vec::new();
        let mut all_hidden_states = Vec::new();

        for i in 0..self.hidden_layers.len() {
            let layer = &mut self.hidden_layers[i];
            let layer_past = &past_key_values[i];

            if self.is_parallel {
                // TODO
            }

            if output_hidden_states {
                all_hidden_states.push(hidden_states.clone());
            }

            let (local_hidden_states, present, attn_weights) = layer.forward(
                &hidden_states,
                &position_ids,
                layer_past,
                &attention_mask,
                &head_mask,
                use_cache,
                output_attentions,
                &device,
            )?;

            hidden_states = local_hidden_states;

            if use_cache {
                presents.push(present.unwrap());
            }

            if output_attentions {
                all_self_attentions.push(attn_weights.unwrap());
            }

            if self.is_parallel {
                // TODO
            }
        }

        hidden_states = self.layernorm_final.forward(&hidden_states)?;

        let output_shape_0 = hidden_states.dims().iter().product::<usize>()
            / partial_output_shape.iter().product::<usize>();

        let mut output_shape = vec![output_shape_0];
        output_shape.append(&mut partial_output_shape);

        hidden_states = hidden_states.reshape(output_shape)?;

        Ok(hidden_states)
    }

    fn get_head_mask(
        head_mask: Option<&Tensor>,
        num_hidden_layers: usize,
        is_attention_chunkced: bool,
        dtype: DType,
    ) -> Option<Tensor> {
        if let Some(head_mask) = head_mask {
            let mut head_mask =
                Self::convert_head_mask_to_5d(head_mask, num_hidden_layers, dtype).unwrap();

            if is_attention_chunkced {
                head_mask = head_mask.unsqueeze(D::Minus1).unwrap();
            }

            Some(head_mask)
        } else {
            None
        }
    }

    fn convert_head_mask_to_5d(
        head_mask: &Tensor,
        num_hidden_layers: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let dim = head_mask.rank();

        let head_mask = if dim == 1 {
            head_mask
                .unsqueeze(0)?
                .unsqueeze(0)?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?
                .expand((num_hidden_layers,))?
        } else if dim == 2 {
            head_mask
                .unsqueeze(1)?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?
        } else {
            head_mask.clone()
        };

        assert_eq!(
            head_mask.rank(),
            5,
            "Num dimension of 'head_mask' must be 5, is {}",
            head_mask.rank()
        );

        head_mask.to_dtype(dtype)
    }
}

impl HiddenLayer {
    pub fn new(
        layer_vb: VarBuilder,
        embed_size: usize,
        inner_size: usize,
        num_heads: usize,
        lm_eps: f64,
        max_pos_embeddings: usize,
        pos_embed_dimension: usize,
        rotary_dim: usize,
        attn_pdrop: f32,
        resid_pdrop: f32,
        device: &Device,
    ) -> HiddenLayer {
        let layer_norm_vb = layer_vb.pp("ln_1");
        let attn_vb = layer_vb.pp("attn");
        let mlp_vb = layer_vb.pp("mlp");

        let layer_norm = LayerNorm::new(
            layer_norm_vb.get(embed_size, "weight").unwrap(),
            layer_norm_vb.get(embed_size, "bias").unwrap(),
            lm_eps,
        );

        let head_size = embed_size / num_heads;

        let attention = Attention::new(
            attn_vb,
            embed_size,
            num_heads,
            head_size,
            max_pos_embeddings,
            pos_embed_dimension,
            rotary_dim,
            attn_pdrop,
            resid_pdrop,
            device,
        );

        let mlp = MLP::new(mlp_vb, inner_size, embed_size, resid_pdrop);

        HiddenLayer {
            layer_norm,
            attention,
            mlp,
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        layer_past: &[Option<Tensor>],
        attention_mask: &Option<Tensor>,
        head_mask: &Option<Tensor>,
        use_cache: bool,
        output_attentions: bool,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let residual = hidden_states;
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        let (attn_output, present, attn_weights) = self.attention.forward(
            &hidden_states,
            position_ids,
            layer_past,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
            device,
        )?;

        let feed_forward_hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (attn_output + feed_forward_hidden_states + residual)?;

        if use_cache {
            Ok((hidden_states, present, attn_weights))
        } else {
            Ok((hidden_states, None, attn_weights))
        }
    }
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        embed_size: usize,
        num_heads: usize,
        head_size: usize,
        max_pos_embeddings: usize,
        pos_embed_dimension: usize,
        rotary_dim: usize,
        attn_pdrop: f32,
        resid_pdrop: f32,
        device: &Device,
    ) -> Attention {
        let q = linear_no_bias(embed_size, embed_size, vb.pp("q_proj")).unwrap();
        let k = linear_no_bias(embed_size, embed_size, vb.pp("k_proj")).unwrap();
        let v = linear_no_bias(embed_size, embed_size, vb.pp("v_proj")).unwrap();
        let out = linear_no_bias(embed_size, embed_size, vb.pp("out_proj")).unwrap();

        let mut bias_vec = vec![0u8; max_pos_embeddings * max_pos_embeddings];

        for i in 0..max_pos_embeddings {
            for j in 0..max_pos_embeddings {
                let is_tril = i <= j;

                if is_tril {
                    bias_vec[i + j * max_pos_embeddings] = 1;
                }
            }
        }

        let bias = Tensor::from_vec(
            bias_vec,
            &[1, 1, max_pos_embeddings, max_pos_embeddings],
            device,
        )
        .unwrap();

        let embed_positions =
            Self::create_sinusoidal_positions(max_pos_embeddings, pos_embed_dimension, device)
                .unwrap();

        let attn_dropout = Dropout::new(attn_pdrop);
        let resid_dropout = Dropout::new(resid_pdrop);

        Attention {
            q,
            k,
            v,
            out,
            num_heads,
            head_size,
            rotary_dim,
            bias,
            embed_positions,
            attn_dropout,
            resid_dropout,
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        layer_past: &[Option<Tensor>],
        attention_mask: &Option<Tensor>,
        head_mask: &Option<Tensor>,
        use_cache: bool,
        output_attentions: bool,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let q = self.q.forward(&hidden_states)?;
        let k = self.k.forward(&hidden_states)?;
        let v = self.v.forward(&hidden_states)?;

        let q = Self::split_heads(&q, self.num_heads, self.head_size, true)?;
        let k = Self::split_heads(&k, self.num_heads, self.head_size, true)?;
        let v = Self::split_heads(&v, self.num_heads, self.head_size, false)?;

        let embed_positions = self.get_embed_positions(position_ids)?;

        let repeated_position_ids = position_ids
            .unsqueeze(D::Minus1)?
            .repeat(&[1, 1, embed_positions.dim(D::Minus1)?])?
            .contiguous()?;

        let sincos = embed_positions.gather(&repeated_position_ids, 1)?;
        let sincos_half_count = sincos.dim(D::Minus1)? / 2;
        let sin = sincos.narrow(D::Minus1, 0, sincos_half_count)?;
        let cos = sincos.narrow(
            D::Minus1,
            sincos_half_count,
            sincos.dim(D::Minus1)? - sincos_half_count,
        )?;

        let (k, q) = if self.rotary_dim == 0 {
            let k = Self::apply_rotary_pos_emb(&k, &sin, &cos)?;
            let q = Self::apply_rotary_pos_emb(&q, &sin, &cos)?;

            (k, q)
        } else {
            let k_rot = k.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(3, self.rotary_dim, k.dim(3)? - self.rotary_dim)?;

            let q_rot = q.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(3, self.rotary_dim, q.dim(3)? - self.rotary_dim)?;

            let k_rot = Self::apply_rotary_pos_emb(&k_rot, &sin, &cos)?;
            let q_rot = Self::apply_rotary_pos_emb(&q_rot, &sin, &cos)?;

            let k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
            let q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;

            (k, q)
        };

        let k = k.permute((0, 2, 1, 3))?;
        let q = q.permute((0, 2, 1, 3))?;

        let (k, v) = if layer_past.len() != 0 {
            let past_key = layer_past.get(0).unwrap().as_ref().unwrap();
            let past_value = layer_past.get(1).unwrap().as_ref().unwrap();

            let k = Tensor::cat(&[past_key, &k], D::Minus2)?;
            let v = Tensor::cat(&[past_value, &v], D::Minus2)?;

            (k, v)
        } else {
            (k, v)
        };

        let k = k.contiguous()?;
        let q = q.contiguous()?;
        let v = v.contiguous()?;

        let present = if use_cache {
            Some(k.to_dtype(hidden_states.dtype())?)
        } else {
            None
        };

        let (attn_output, attn_weights) =
            self.get_attention(&q, &k, &v, attention_mask, head_mask)?;

        let attn_output = Self::merge_heads(&attn_output, self.num_heads, self.head_size)?;
        let attn_output = self.out.forward(&attn_output)?;
        let attn_output = self.resid_dropout.forward(&attn_output, false)?;

        if output_attentions {
            Ok((attn_output, present, Some(attn_weights)))
        } else {
            Ok((attn_output, present, None))
        }
    }

    fn merge_heads(attn_output: &Tensor, num_heads: usize, head_size: usize) -> Result<Tensor> {
        let num_dims = attn_output.dims().len();

        let attn_output = if num_dims == 5 {
            attn_output.permute((0, 1, 3, 2, 4))?.contiguous()?
        } else if num_dims == 4 {
            attn_output.permute((0, 2, 1, 3))?.contiguous()?
        } else {
            panic!(
                "Input tensor rank should be one of [4, 5], but is: {}",
                num_dims
            )
        };

        let mut new_shape = attn_output.dims()[0..attn_output.dims().len() - 2].to_vec();
        new_shape.push(num_heads * head_size);

        attn_output.reshape(new_shape)
    }

    fn get_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: &Option<Tensor>,
        head_mask: &Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let query = query.to_dtype(candle_core::DType::F32)?;
        let key = key.to_dtype(candle_core::DType::F32)?;

        let attn_weights = query.matmul(&key.transpose(D::Minus1, D::Minus2)?)?;

        let query_length = query.dim(D::Minus2)?;
        let key_length = key.dim(D::Minus2)?;

        let causal_mask = self
            .bias
            .narrow(2, key_length - query_length, key_length)?
            .narrow(3, 0, key_length)?
            .broadcast_as(attn_weights.shape())?;

        let mask_value =
            Tensor::new(&[f32::MIN], attn_weights.device())?.broadcast_as(attn_weights.shape())?;

        let mut attn_weights = causal_mask.where_cond(&attn_weights, &mask_value)?;

        if let Some(attention_mask) = attention_mask {
            attn_weights = (attn_weights + attention_mask)?;
        }

        let attn_weights = softmax(&attn_weights, D::Minus1)?;
        let attn_weights = attn_weights.to_dtype(value.dtype())?;
        let mut attn_weights = self.attn_dropout.forward(&attn_weights, false)?;

        if let Some(head_mask) = head_mask {
            attn_weights = (attn_weights * head_mask)?
        }

        let attn_output = attn_weights.matmul(&value)?;

        Ok((attn_output, attn_weights))
    }

    fn create_sinusoidal_positions(
        num_pos: usize,
        dimension: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let inv_freq = (0..dimension)
            .step_by(2)
            .map(|x| 1.0 / (10000f32.powf(x as f32 / dimension as f32)))
            .collect::<Vec<_>>();

        let pos_ids = Tensor::arange(0f32, num_pos as f32, device)?.unsqueeze(1)?;
        let inv_freq = Tensor::new(inv_freq, device)?.unsqueeze(0)?;

        let sinusoid_inp = pos_ids.matmul(&inv_freq)?;

        let sin = Tensor::sin(&sinusoid_inp)?;
        let cos = Tensor::cos(&sinusoid_inp)?;

        Tensor::cat(&[sin, cos], 1)
    }

    fn split_heads(
        input: &Tensor,
        num_heads: usize,
        head_size: usize,
        do_rotary: bool,
    ) -> Result<Tensor> {
        let new_shape = &input.dims()[0..input.dims().len() - 1];

        let new_shape = new_shape
            .iter()
            .chain(&[num_heads, head_size])
            .map(|x| *x)
            .collect::<Vec<_>>();

        let new_shape_dim = new_shape.len();

        let ret = input.reshape(new_shape)?;

        if do_rotary {
            Ok(ret)
        } else if new_shape_dim == 5 {
            ret.permute((0, 1, 3, 2, 4))
        } else if new_shape_dim == 4 {
            ret.permute((0, 2, 1, 3))
        } else {
            panic!("Invalid shape")
        }
    }

    fn get_embed_positions(&mut self, input: &Tensor) -> Result<Tensor> {
        self.embed_positions = self.embed_positions.to_device(input.device())?;

        self.embed_positions.repeat((input.shape().dims()[0], 1, 1))
    }

    fn apply_rotary_pos_emb(tensor: &Tensor, sin: &Tensor, cos: &Tensor) -> Result<Tensor> {
        let repeats = 2;
        let dim = 2;

        let sin = repeat_interleave(&sin, repeats, dim)?;
        let cos = repeat_interleave(&cos, repeats, dim)?;

        let rotated = Self::rotate_every_two(tensor)?;

        tensor.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)
    }

    fn rotate_every_two(tensor: &Tensor) -> Result<Tensor> {
        let rotate_dim = 3;
        let dim_count = tensor.dim(rotate_dim)? as u32;

        let zero_start_indices = Tensor::arange_step(0, dim_count, 2, tensor.device())?;
        let one_start_indices = Tensor::arange_step(1, dim_count, 2, tensor.device())?;

        let x_zero = tensor.index_select(&zero_start_indices, rotate_dim)?;
        let x_one = tensor.index_select(&one_start_indices, rotate_dim)?;

        let x = Tensor::stack(&[x_one.neg()?, x_zero], D::Minus1)?;

        x.flatten(D::Minus2, D::Minus1)
    }
}

impl MLP {
    pub fn new(vb: VarBuilder, inter_size: usize, embed_size: usize, resid_pdrop: f32) -> MLP {
        let fc_in = linear(embed_size, inter_size, vb.pp("fc_in")).unwrap();
        let fc_out = linear(inter_size, embed_size, vb.pp("fc_out")).unwrap();

        let activation = candle_nn::activation::Activation::NewGelu;
        let dropout = Dropout::new(resid_pdrop);

        MLP {
            fc_in,
            fc_out,
            activation,
            dropout,
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = self.fc_in.forward(&input)?;
        let input = self.activation.forward(&input)?;
        let input = self.fc_out.forward(&input)?;
        let input = self.dropout.forward(&input, false)?;

        Ok(input)
    }
}

fn repeat_interleave(tensor: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let tensor = tensor.unsqueeze(dim)?;

    let mut shape = tensor.dims().to_vec();
    shape[dim] = repeats;

    let tensor = tensor.broadcast_as(shape)?;
    let tensor = tensor.transpose(dim, dim + 1)?;
    let tensor = tensor.flatten(dim, dim + 1)?;
    let tensor = tensor.unsqueeze(dim);

    tensor
}
