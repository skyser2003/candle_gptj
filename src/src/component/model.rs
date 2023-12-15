use std::{fs::File, path::PathBuf};

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{ops::softmax, Dropout, Embedding, LayerNorm, Linear, Module, VarBuilder};
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

        Self {
            word_token_embedding,
            hidden_layers: layers,
            layernorm_final,
        }
    }

    fn forward(
        &mut self,
        input_ids: &Vec<i64>,
        position_ids: Option<&Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let input_ids = Tensor::new(input_ids.clone(), device)?;

        let input_ids = self.word_token_embedding.forward(&input_ids)?;

        let mut hidden_state = input_ids;

        for layer in &mut self.hidden_layers {
            hidden_state = layer.forward(
                &hidden_state,
                position_ids.unwrap(),
                None,
                None,
                None,
                false,
                false,
                device,
            )?;
        }

        hidden_state = self.layernorm_final.forward(&hidden_state)?;

        Ok(hidden_state)
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

        let mlp = MLP::new(mlp_vb, inner_size, embed_size);

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
        layer_past: Option<&[&Tensor]>,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
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

        let feed_forward_hidden_states = self.mlp.forward(&attn_output)?;
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

        let mut bias_vec = vec![0u8; max_pos_embeddings * max_pos_embeddings];

        for i in 0..max_pos_embeddings {
            for j in 0..max_pos_embeddings {
                let is_tril = j <= i;

                if is_tril {
                    bias_vec[i + j * max_pos_embeddings] = 1;
                }
            }
        }

        let bias =
            Tensor::from_vec(bias_vec, &[max_pos_embeddings, max_pos_embeddings], device).unwrap();

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
        layer_past: Option<&[&Tensor]>,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
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

        let repeated_position_ids =
            position_ids
                .unsqueeze(D::Minus1)?
                .repeat(&[1, 1, embed_positions.dim(D::Minus1)?])?;

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
            let k_rot = k.narrow(3, 0, self.rotary_dim)?;
            let k_pass = k.narrow(3, self.rotary_dim, k.dim(3)? - self.rotary_dim)?;

            let q_rot = q.narrow(3, 0, self.rotary_dim)?;
            let q_pass = q.narrow(3, self.rotary_dim, q.dim(3)? - self.rotary_dim)?;

            let k_rot = Self::apply_rotary_pos_emb(&k_rot, &sin, &cos)?;
            let q_rot = Self::apply_rotary_pos_emb(&q_rot, &sin, &cos)?;

            let k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
            let q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;

            (k, q)
        };

        let k = k.permute((0, 2, 1, 3))?;
        let q = q.permute((0, 2, 1, 3))?;

        let (k, v) = if let Some(layer_past) = layer_past {
            let past_key = layer_past[0];
            let past_value = layer_past[1];

            let k = Tensor::cat(&[past_key, &k], D::Minus2)?;
            let v = Tensor::cat(&[past_value, &v], D::Minus2)?;

            (k, v)
        } else {
            (k, v)
        };

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

        let merged = if num_dims == 5 {
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

        merged.reshape(new_shape)
    }

    fn get_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let query_length = query.dim(D::Minus2)?;
        let key_length = key.dim(D::Minus2)?;

        let causal_mask = self
            .bias
            .narrow(2, key_length - query_length, key_length)?
            .narrow(3, 0, key_length)?;

        let query = query.to_dtype(candle_core::DType::F32)?;
        let key = key.to_dtype(candle_core::DType::F32)?;

        let attn_weights = query.matmul(&key.transpose(D::Minus1, D::Minus2)?)?;
        let mask_value = Tensor::new(&[f32::MIN], attn_weights.device())?;
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
        let new_shape = &input.dims()[0..input.dim(D::Minus1)?];
        let new_shape = new_shape
            .iter()
            .chain(&[num_heads, head_size])
            .map(|x| *x)
            .collect::<Vec<_>>();

        let ret = input.reshape(new_shape.clone())?;

        if do_rotary {
            Ok(ret)
        } else if new_shape.len() == 5 {
            ret.permute((0, 1, 3, 2, 4))
        } else if new_shape.len() == 4 {
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
        let repeat_dim = 2;
        let repeat_count = 3;

        let sin = sin.unsqueeze(repeat_dim)?;
        let cos = cos.unsqueeze(repeat_dim)?;

        let mut dim_sin = sin.dims().to_vec();
        let mut dim_cos = cos.dims().to_vec();

        dim_sin[repeat_dim] = repeat_count;
        dim_cos[repeat_dim] = repeat_count;

        let sin = sin.broadcast_as(dim_sin)?;
        let cos = cos.broadcast_as(dim_cos)?;

        (tensor * cos)? + (Self::rotate_every_two(tensor)? * sin)
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

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = self.fc_in.forward(&input)?;
        let input = input.gelu()?;
        let input = self.fc_out.forward(&input)?;

        Ok(input)
    }
}
