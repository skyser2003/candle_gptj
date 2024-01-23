use std::ops::Mul;
use std::time::Instant;
use std::{collections::HashMap, fs::File, path::PathBuf};

use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use tch::nn::{
    self, embedding, layer_norm, linear, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    Linear, LinearConfig, Module, VarStore,
};
use tch::{Device, IndexOp, Kind, Result, TchError, Tensor};
use tokenizers::Tokenizer;

use super::model_base::GPTJConfig;

pub struct ModelLoader {
    model: CausalModel,
    tokenizer: Tokenizer,
}

pub struct CausalModel {
    pub buffer: memmap2::Mmap,
    pub model_filename: PathBuf,
    pub device: Device,
    pub config: GPTJConfig,
    pub transformer: CoreModel,
    pub lm_head: Linear,
}

pub struct CoreModel {
    pub config: GPTJConfig,
    pub dtype: Kind,
    pub dtype_min: f32,
    pub word_token_embedding: Embedding,
    pub hidden_layers: Vec<HiddenLayer>,
    pub layernorm_final: LayerNorm,
    pub is_parallel: bool,
}

pub struct HiddenLayer {
    pub layer_norm: LayerNorm,
    pub attention: Attention,
    pub mlp: MLP,
}

pub struct Attention {
    pub qkv: Option<Linear>,
    pub out: Linear,
    pub num_heads: usize,
    pub head_size: usize,
    pub rotary_dim: usize,
    pub embed_size: usize,
    pub scale_attn: Tensor,
    pub bias: Tensor,
    pub embed_positions: Tensor,
    pub attn_pdrop: f32,
    pub resid_pdrop: f32,
}

pub struct MLP {
    pub fc_in: Linear,
    pub fc_out: Linear,
    pub resid_pdrop: f32,
}

pub struct CausalOutput {
    pub last_hidden_state: Tensor,
    pub past_key_values: Vec<(Tensor, Tensor)>,
    pub hidden_states: Vec<Tensor>,
    pub attentions: Vec<Tensor>,
}

trait LogitsWarper {
    fn process(&self, input_ids: &Tensor, base_logits: &Tensor) -> Tensor;
}

struct TopKLogitsWarper {
    k: i64,
}

struct TopPLogitsWarper {
    p: f64,
    min_tokens: i64,
}

pub struct GenerationConfig {
    top_k: Option<i64>,
    top_p: Option<f64>,
    max_tokens: Option<i32>,
}

impl LogitsWarper for TopKLogitsWarper {
    fn process(&self, _: &Tensor, base_logits: &Tensor) -> Tensor {
        let (top_k_logits, _) = base_logits.topk(self.k, -1, true, false);
        let top_k_below = &top_k_logits.narrow(-1, top_k_logits.size()[1] - 1, 1);
        let remove_indices = base_logits.less_tensor(&top_k_below);

        let base_logits = base_logits.masked_fill(&remove_indices, 0);

        base_logits.softmax(-1, Kind::Float).multinomial(1, false)
    }
}

impl LogitsWarper for TopPLogitsWarper {
    fn process(&self, _: &Tensor, base_logits: &Tensor) -> Tensor {
        let (logits, sorted_indices) = base_logits.sort(-1, true);
        let logits = logits.softmax(-1, Kind::Float);
        let cumsum_logits = logits.cumsum(-1, Kind::Float);

        let top_p_below = cumsum_logits.ge(self.p);
        let _ = top_p_below.i((.., 0..self.min_tokens)).fill_(0);

        let remove_indices = top_p_below.scatter(1, &sorted_indices, &top_p_below);

        base_logits.masked_fill(&remove_indices, 0)
    }
}

impl ModelLoader {
    pub fn new(
        model_dir: &str,
        tokenizer_dir: &str,
        dtype: Option<String>,
        device: &Device,
    ) -> ModelLoader {
        println!("Begin loading model...");
        let start_time = Instant::now();

        let model_dir = std::path::Path::new(model_dir);

        let config_filename = model_dir.join("config.json");
        let config: GPTJConfig =
            serde_json::from_reader(File::open(config_filename).unwrap()).unwrap();

        let model_filename = model_dir.join("model.safetensors");
        let buffer = Self::load_model(&model_filename);

        let dtype_map = vec![
            ("float16", (Kind::Half, -65504.0)),
            ("float32", (Kind::Float, f32::MIN)),
            ("bfloat16", (Kind::BFloat16, -3.38e38)),
        ]
        .into_iter()
        .collect::<HashMap<_, _>>();

        let torch_dtype = dtype.unwrap_or(config.torch_dtype.clone().unwrap_or("".to_string()));

        let torch_dtype = if dtype_map.contains_key(torch_dtype.as_str()) {
            torch_dtype
        } else {
            "float32".to_string()
        };

        let (dtype, dtype_min) = *dtype_map.get(torch_dtype.as_str()).unwrap();

        let mut vs = VarStore::new(*device);
        let vr = &vs.root();

        let mut core_model = CoreModel::new(vr, &dtype, dtype_min, &device, &config);
        let lm_head = linear(
            vr / "lm_head",
            config.n_embd as i64,
            config.vocab_size as i64,
            LinearConfig {
                ..Default::default()
            },
        );

        vs.load(model_filename.clone()).unwrap();
        vs.set_kind(dtype);

        let vb = SafeTensors::deserialize(&buffer);
        core_model.post_load(&vb.unwrap(), dtype, *device);

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

        let _ = instance.inference(&["Hot loading"], None);

        let end_time = Instant::now();

        println!(
            "Loading model done, dtype={:?}, {}s",
            dtype,
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

    pub fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        past_key_values: Option<Vec<(&Tensor, &Tensor)>>,
    ) -> Result<CausalOutput> {
        let mut output = self.model.transformer.forward(
            input_ids,
            past_key_values,
            None,
            None,
            None,
            None,
            input_embeds,
            None,
            false,
            false,
        )?;

        let lm_logits = self
            .model
            .lm_head
            .forward(&output.last_hidden_state)
            .to_kind(Kind::Float);

        output.last_hidden_state = lm_logits;

        Ok(output)
    }

    pub fn inference(
        &mut self,
        inputs: &[&str],
        config: Option<GenerationConfig>,
    ) -> Result<Vec<String>> {
        let config = config.unwrap_or(GenerationConfig {
            max_tokens: Some(50),
            top_k: Some(1),
            top_p: Some(0.0),
        });

        let encodings = self.tokenizer.encode_batch(inputs.to_vec(), true).unwrap();
        let input_tokens = encodings
            .iter()
            .map(|enc| {
                enc.get_ids()
                    .iter()
                    .map(|val| *val as i64)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        let input_ids = Tensor::from_slice(&input_tokens)
            .reshape([encodings.len() as i64, -1])
            .to_device(self.model.device);

        let mut embeds = self.model.transformer.create_embed(&input_ids);

        let mut gen_tokens = vec![vec![]; inputs.len()];
        let mut past_key_values: Option<Vec<(Tensor, Tensor)>> = None;

        for _ in 0..config.max_tokens.unwrap() {
            let opt_past_key_values = if let Some(past_key_values) = &past_key_values {
                Some(
                    past_key_values
                        .iter()
                        .map(|(key, value)| (key, value))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };

            let causal_output = self.forward(None, Some(embeds.copy()), opt_past_key_values)?;

            let all_logits = causal_output.last_hidden_state;
            past_key_values = Some(causal_output.past_key_values);

            let logits = all_logits.narrow(1, -1, 1);

            // Post processing
            let is_greedy = false;

            let top_p_warper = TopPLogitsWarper {
                p: config.top_p.unwrap(),
                min_tokens: 1,
            };

            let top_k_warper = TopKLogitsWarper {
                k: config.top_k.unwrap(),
            };

            let indices = if is_greedy {
                logits.argmax(-1, false)
            } else {
                let mut logits_shape = logits.size();
                logits_shape.pop();

                let base_logits = logits.reshape([-1, *logits.size().last().unwrap() as i64]);

                let base_logits = top_p_warper.process(&input_ids, &base_logits);
                let base_logits = top_k_warper.process(&input_ids, &base_logits);

                base_logits.narrow(-1, 0, 1).reshape(logits_shape)
            };

            let gen_embeds = self.model.transformer.create_embed(&indices);

            embeds = gen_embeds;

            for (tokens, index) in gen_tokens
                .iter_mut()
                .zip(&Vec::<Vec<i64>>::try_from(indices).unwrap())
            {
                tokens.push(index[0]);
            }
        }

        let indices = gen_tokens;
        let indices = indices
            .iter()
            .map(|nested_vec| {
                nested_vec
                    .into_iter()
                    .map(|token| *token as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let indices = indices
            .iter()
            .map(|nested_vec| nested_vec.as_slice())
            .collect::<Vec<_>>();

        let outputs = self.tokenizer.decode_batch(&indices, true).unwrap();

        return Ok(outputs);
    }

    pub fn get_tensors(&self) -> SafeTensors {
        self.model.tensors()
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn get_config(&self) -> &GPTJConfig {
        &self.model.config
    }
}

impl CausalModel {
    fn tensors(&self) -> SafeTensors {
        let tensors = SafeTensors::deserialize(&self.buffer).unwrap();

        return tensors;
    }
}

impl CoreModel {
    fn new(
        vr: &nn::Path,
        dtype: &Kind,
        dtype_min: f32,
        device: &Device,
        config: &GPTJConfig,
    ) -> CoreModel {
        let tf_vb = &(vr / ("transformer"));

        let word_token_embedding = embedding(
            tf_vb / "wte",
            config.vocab_size as i64,
            config.n_embd as i64,
            EmbeddingConfig {
                ..Default::default()
            },
        );

        let hidden_layers_vb = tf_vb / "h";
        let inner_size = config.n_inner.unwrap_or(4 * config.n_embd);
        let rotary_dim = config.rotary_dim.unwrap_or(0);
        let pos_embed_dimension = if rotary_dim == 0 {
            config.n_embd
        } else {
            rotary_dim
        };

        let layers = (0..config.n_layer)
            .map(|i| {
                let layer_vb = &hidden_layers_vb / i;
                let layer = HiddenLayer::new(
                    &layer_vb,
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

        let layernorm_final = layer_norm(
            tf_vb / "ln_f",
            vec![config.n_embd as i64],
            nn::LayerNormConfig {
                cudnn_enabled: device.is_cuda(),
                eps: config.layer_norm_epsilon,
                ..Default::default()
            },
        );

        Self {
            config: config.clone(),
            dtype: *dtype,
            dtype_min,
            word_token_embedding,
            hidden_layers: layers,
            layernorm_final,
            is_parallel: false,
        }
    }

    fn post_load(&mut self, buffer: &SafeTensors, kind: Kind, device: Device) {
        for (i, ele) in self.hidden_layers.iter_mut().enumerate() {
            ele.post_load(i, buffer, kind, device);
        }
    }

    fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        past_key_values: Option<Vec<(&Tensor, &Tensor)>>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<Tensor>,
        head_mask: Option<&Tensor>,
        input_embeds: Option<Tensor>,
        use_cache: Option<bool>,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> Result<CausalOutput> {
        let use_cache = use_cache.unwrap_or(self.config.use_cache);

        let (input_shape, batch_size, device) = match (&input_ids, &input_embeds) {
            (Some(_), Some(_)) => {
                panic!("You cannot specify both input_ids and inputs_embeds at the same time")
            }
            (Some(input_ids), None) => {
                // TODO
                // self.warn_if_padding_and_no_attention_mask(&input_ids, attention_mask);

                let device = input_ids.device();

                let input_shape = input_ids.size();
                let input_ids = input_ids.reshape([-1, *input_shape.last().unwrap()]);
                let batch_size = input_ids.size()[0];

                (input_shape, batch_size, device)
            }
            (None, Some(input_embeds)) => {
                let device = input_embeds.device();

                let input_shape = input_embeds.size();
                let input_shape = Vec::from(&input_shape[0..input_shape.len() - 1]);
                let batch_size = input_shape[0];

                (input_shape, batch_size, device)
            }
            (None, None) => panic!("You have to specify either input_ids or inputs_embeds"),
        };

        let token_type_ids = if let Some(token_type_ids) = token_type_ids {
            Some(token_type_ids.reshape([-1, *token_type_ids.size().last().unwrap()]))
        } else {
            None
        };

        let (past_length, past_key_values) = if let Some(past_key_values) = past_key_values {
            let past_length = *past_key_values[0].0.size().last().unwrap();

            (
                past_length,
                past_key_values
                    .into_iter()
                    .map(|layers_past| Some(layers_past))
                    .collect::<Vec<_>>(),
            )
        } else {
            let past_length = 0i64;
            let past_key_values = vec![None; self.config.n_layer];

            (past_length, past_key_values)
        };

        let position_ids = if position_ids.is_none() {
            let position_ids = Tensor::arange_start(
                past_length as i64,
                *input_shape.last().unwrap() + past_length,
                (Kind::Int64, device),
            );
            position_ids.unsqueeze(0)
        } else {
            position_ids.unwrap()
        };

        let attention_mask = if let Some(attention_mask) = attention_mask {
            if batch_size <= 0 {
                panic!("Batch size has to be defined and > 0");
            }

            let attention_mask = attention_mask.reshape([batch_size, -1]);
            let attention_mask = attention_mask.unsqueeze(1).unsqueeze(1);
            let attention_mask = attention_mask.to_kind(self.dtype);
            let attention_mask = (1.0 - attention_mask) * self.dtype_min as f64;

            Some(attention_mask)
        } else {
            None
        };

        let head_mask = Self::get_head_mask(head_mask, self.config.n_layer, false, self.dtype);

        let input_embeds = if let Some(input_embeds) = input_embeds {
            input_embeds
        } else {
            self.create_embed(input_ids.unwrap())
        };

        let mut hidden_states = input_embeds;

        if let Some(token_type_ids) = token_type_ids {
            let token_type_embeds = self.word_token_embedding.forward(&token_type_ids);
            hidden_states = hidden_states + token_type_embeds;
        }

        let _ = hidden_states.dropout_(self.config.embd_pdrop as f64, false);

        let mut partial_output_shape = Vec::new();

        for i in input_shape.iter().skip(1) {
            partial_output_shape.push(*i);
        }

        partial_output_shape.push(*hidden_states.size().last().unwrap());

        let mut presents = Vec::new();
        let mut all_self_attentions = Vec::new();
        let mut all_hidden_states = Vec::new();

        for i in 0..self.hidden_layers.len() {
            let layer = &mut self.hidden_layers[i];
            let layer_past = past_key_values[i];

            if self.is_parallel {
                // TODO
            }

            if output_hidden_states {
                all_hidden_states.push(hidden_states.shallow_clone());
            }

            let (local_hidden_states, present, attn_weights) = layer.forward(
                &hidden_states,
                &position_ids,
                layer_past,
                &attention_mask,
                &head_mask,
                use_cache,
                output_attentions,
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

        hidden_states = self.layernorm_final.forward(&hidden_states);

        let output_shape_0 = hidden_states.size().iter().product::<i64>()
            / partial_output_shape.iter().product::<i64>();

        let mut output_shape = vec![output_shape_0];
        output_shape.append(&mut partial_output_shape);

        hidden_states = hidden_states.reshape(output_shape);

        let output = CausalOutput {
            last_hidden_state: hidden_states,
            past_key_values: presents,
            hidden_states: all_hidden_states,
            attentions: all_self_attentions,
        };

        Ok(output)
    }

    pub fn create_embed(&self, input_ids: &Tensor) -> Tensor {
        self.word_token_embedding.forward(input_ids)
    }

    fn get_head_mask(
        head_mask: Option<&Tensor>,
        num_hidden_layers: usize,
        is_attention_chunkced: bool,
        dtype: Kind,
    ) -> Option<Tensor> {
        if let Some(head_mask) = head_mask {
            let mut head_mask =
                Self::convert_head_mask_to_5d(head_mask, num_hidden_layers as i64, dtype);

            if is_attention_chunkced {
                head_mask = head_mask.unsqueeze(-1);
            }

            Some(head_mask)
        } else {
            None
        }
    }

    fn convert_head_mask_to_5d(head_mask: &Tensor, num_hidden_layers: i64, dtype: Kind) -> Tensor {
        let dim = head_mask.dim();

        let head_mask = if dim == 1 {
            head_mask
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand([num_hidden_layers], false)
        } else if dim == 2 {
            head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        } else {
            head_mask.shallow_clone()
        };

        assert_eq!(
            head_mask.dim(),
            5,
            "Num dimension of 'head_mask' must be 5, is {}",
            head_mask.dim()
        );

        head_mask.to_kind(dtype)
    }
}

impl HiddenLayer {
    pub fn new(
        layer_vb: &nn::Path,
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
        let layer_norm_vb = layer_vb / "ln_1";
        let attn_vb = layer_vb / "attn";
        let mlp_vb = layer_vb / "mlp";

        let layer_norm = layer_norm(
            layer_norm_vb,
            [embed_size as i64].to_vec(),
            LayerNormConfig {
                cudnn_enabled: device.is_cuda(),
                eps: lm_eps,
                ..Default::default()
            },
        );

        let head_size = embed_size / num_heads;

        let attention = Attention::new(
            &attn_vb,
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

        let mlp = MLP::new(&mlp_vb, inner_size, embed_size, resid_pdrop);

        HiddenLayer {
            layer_norm,
            attention,
            mlp,
        }
    }

    fn post_load(&mut self, layer_number: usize, buffer: &SafeTensors, kind: Kind, device: Device) {
        self.attention.post_load(layer_number, buffer, kind, device);
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        layer_past: Option<(&Tensor, &Tensor)>,
        attention_mask: &Option<Tensor>,
        head_mask: &Option<Tensor>,
        use_cache: bool,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>, Option<Tensor>)> {
        let residual = hidden_states;
        let hidden_states = self.layer_norm.forward(&hidden_states);
        let (attn_output, present, attn_weights) = self.attention.forward(
            &hidden_states,
            position_ids,
            layer_past,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
        )?;

        let feed_forward_hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = attn_output + feed_forward_hidden_states + residual;

        if use_cache {
            Ok((hidden_states, present, attn_weights))
        } else {
            Ok((hidden_states, None, attn_weights))
        }
    }
}

impl Attention {
    pub fn new(
        vb: &nn::Path,
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
        let embed_shape = [embed_size as i64, embed_size as i64];

        let out = Linear {
            ws: (vb / "out_proj").var("weight", &embed_shape, nn::Init::Const(0.0)),
            bs: None,
        };

        let mut bias_vec = vec![false; max_pos_embeddings * max_pos_embeddings];

        for i in 0..max_pos_embeddings {
            for j in 0..max_pos_embeddings {
                let is_tril = i <= j;

                if is_tril {
                    bias_vec[i + j * max_pos_embeddings] = true;
                }
            }
        }

        let bias = Tensor::from_slice(&bias_vec)
            .reshape([
                1i64,
                1,
                max_pos_embeddings as i64,
                max_pos_embeddings as i64,
            ])
            .to_device(*device);

        let mut scale_attn = Tensor::from_slice(&[head_size as f32]).to_device(*device);
        scale_attn.sqrt_();

        let embed_positions = Self::create_sinusoidal_positions(
            max_pos_embeddings,
            pos_embed_dimension,
            device,
            out.ws.kind(),
        );

        Attention {
            qkv: None,
            out,
            num_heads,
            head_size,
            rotary_dim,
            embed_size,
            bias,
            scale_attn,
            embed_positions,
            attn_pdrop,
            resid_pdrop,
        }
    }

    fn tensor_from_buffer(view: TensorView<'_>) -> core::result::Result<Tensor, TchError> {
        let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
        let dtype = view.dtype();
        let kind: Kind = match dtype {
            safetensors::Dtype::F32 => Kind::Float,
            safetensors::Dtype::F16 => Kind::Half,
            safetensors::Dtype::BF16 => Kind::BFloat16,
            _ => unreachable!(),
        };
        Tensor::f_from_data_size(view.data(), &size, kind)
    }

    fn post_load(&mut self, layer_number: usize, buffer: &SafeTensors, kind: Kind, device: Device) {
        let prefix = format!("transformer.h.{}.attn", layer_number);

        let q_proj_weight = Self::tensor_from_buffer(
            buffer
                .tensor(format!("{}.q_proj.weight", prefix).as_str())
                .unwrap(),
        )
        .unwrap();

        let k_proj_weight = Self::tensor_from_buffer(
            buffer
                .tensor(format!("{}.k_proj.weight", prefix).as_str())
                .unwrap(),
        )
        .unwrap();

        let v_proj_weight = Self::tensor_from_buffer(
            buffer
                .tensor(format!("{}.v_proj.weight", prefix).as_str())
                .unwrap(),
        )
        .unwrap();

        let qkv_weight = Tensor::cat(&[q_proj_weight, k_proj_weight, v_proj_weight], 0)
            .to_kind(kind)
            .to_device(device);

        let qkv = Linear {
            ws: qkv_weight,
            bs: None,
        };

        self.qkv = Some(qkv);
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        layer_past: Option<(&Tensor, &Tensor)>,
        attention_mask: &Option<Tensor>,
        head_mask: &Option<Tensor>,
        use_cache: bool,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>, Option<Tensor>)> {
        let qkv = self.qkv.as_ref().unwrap().forward(&hidden_states);

        let qkv = qkv.split(self.embed_size as i64 * 2, -1);
        let qk = &qkv[0];
        let v = &qkv[1];

        let qk = Self::split_heads(qk, self.num_heads * 2, self.head_size, true);
        let v = Self::split_heads(v, self.num_heads, self.head_size, false);

        let embed_positions = self.get_embed_positions(position_ids);

        let repeated_position_ids =
            position_ids
                .unsqueeze(-1)
                .repeat(&[1, 1, *embed_positions.size().last().unwrap()]);

        let sincos = embed_positions.gather(1, &repeated_position_ids, false);
        let sincos_half_count = sincos.size().last().unwrap() / 2;
        let sin = sincos.narrow(-1, 0, sincos_half_count);
        let cos = sincos.narrow(-1, sincos_half_count, sincos_half_count);

        let qk = if self.rotary_dim == 0 {
            let qk = Self::apply_rotary_pos_emb(&qk, &sin, &cos);

            qk
        } else {
            let dim_index = 3;
            let qk_rot = qk.narrow(dim_index, 0, self.rotary_dim as i64).contiguous();
            let qk_pass = qk.narrow(
                dim_index,
                self.rotary_dim as i64,
                qk.size()[dim_index as usize] - self.rotary_dim as i64,
            );

            let qk_rot = Self::apply_rotary_pos_emb(&qk_rot, &sin, &cos);

            let qk = Tensor::cat(&[qk_rot, qk_pass], -1);

            qk
        };

        let qk = qk.permute([0, 2, 1, 3]);

        let qk_dim = 1i64;
        let qk_size = qk.size()[qk_dim as usize] / 2;

        let q = qk.narrow(qk_dim, 0, qk_size);
        let k = qk.narrow(qk_dim, qk_size, qk_size);

        let (k, v) = if let Some(layer_past) = layer_past {
            let past_key = layer_past.0;
            let past_value = layer_past.1;

            println!("{:?}", past_key);
            println!("{:?}", past_value);
            println!("{:?}", k);
            println!("{:?}", v);
            println!("");

            let k = Tensor::cat(&[past_key, &k], -2);
            let v = Tensor::cat(&[past_value, &v], -2);

            (k, v)
        } else {
            (k, v)
        };

        let k = k.contiguous();
        let q = q.contiguous();
        let v = v.contiguous();

        let (attn_output, attn_weights) =
            self.get_attention(&q, &k, &v, attention_mask, head_mask)?;

        let attn_output = Self::merge_heads(&attn_output, self.num_heads, self.head_size);
        let mut attn_output = self.out.forward(&attn_output);
        let _ = attn_output.dropout_(self.resid_pdrop as f64, false);

        let present = if use_cache {
            Some((k.to_kind(hidden_states.kind()), v))
        } else {
            None
        };

        if output_attentions {
            Ok((attn_output, present, Some(attn_weights)))
        } else {
            Ok((attn_output, present, None))
        }
    }

    fn merge_heads(attn_output: &Tensor, num_heads: usize, head_size: usize) -> Tensor {
        let num_dims = attn_output.dim();

        let attn_output = if num_dims == 5 {
            attn_output.permute([0, 1, 3, 2, 4])
        } else if num_dims == 4 {
            attn_output.permute([0, 2, 1, 3])
        } else {
            panic!(
                "Input tensor rank should be one of [4, 5], but is: {}",
                num_dims
            )
        };

        let mut new_shape = attn_output.size()[0..attn_output.size().len() - 2].to_vec();
        new_shape.push((num_heads * head_size) as i64);

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
        let query_length = query.size()[query.size().len() - 2];
        let key_length = key.size()[key.size().len() - 2];

        let causal_mask = self
            .bias
            .narrow(2, key_length - query_length, key_length)
            .narrow(3, 0, key_length);

        let query = query.to_kind(Kind::Float);
        let key = key.to_kind(Kind::Float);

        let attn_weights = query.matmul(&key.transpose(-1, -2));

        let mask_value = Tensor::from_slice(&[f32::MIN]).to_device(attn_weights.device());
        //.broadcast_to(attn_weights.size());

        let mut attn_weights = attn_weights.where_self(&causal_mask, &mask_value);
        let _ = attn_weights.divide_(&self.scale_attn);

        if let Some(attention_mask) = attention_mask {
            attn_weights = (attn_weights + attention_mask);
        }

        let mut attn_weights = attn_weights.softmax(-1, value.kind());
        let _ = attn_weights.dropout_(self.attn_pdrop as f64, false);

        if let Some(head_mask) = head_mask {
            attn_weights = attn_weights.mul(head_mask)
        }

        let attn_output = attn_weights.matmul(&value);

        Ok((attn_output, attn_weights))
    }

    fn create_sinusoidal_positions(
        num_pos: usize,
        dimension: usize,
        device: &Device,
        dtype: Kind,
    ) -> Tensor {
        let inv_freq = (0..dimension)
            .step_by(2)
            .map(|x| 1.0 / (10000f32.powf(x as f32 / dimension as f32)))
            .collect::<Vec<_>>();

        let pos_ids = Tensor::arange(num_pos as i64, (dtype, *device)).unsqueeze(1);
        let inv_freq = Tensor::from_slice(&inv_freq)
            .unsqueeze(0)
            .to_device(*device)
            .to_kind(dtype);

        let sinusoid_inp = pos_ids.matmul(&inv_freq);

        let sin = Tensor::sin(&sinusoid_inp);
        let cos = Tensor::cos(&sinusoid_inp);

        Tensor::cat(&[sin, cos], 1)
    }

    fn split_heads(input: &Tensor, num_heads: usize, head_size: usize, do_rotary: bool) -> Tensor {
        let new_shape = &input.size()[0..input.size().len() - 1];

        let new_shape = new_shape
            .iter()
            .chain(&[num_heads as i64, head_size as i64])
            .map(|x| *x)
            .collect::<Vec<_>>();

        let new_shape_dim = new_shape.len();

        let ret = input.reshape(new_shape);

        if do_rotary {
            ret
        } else if new_shape_dim == 5 {
            ret.permute([0, 1, 3, 2, 4])
        } else if new_shape_dim == 4 {
            ret.permute([0, 2, 1, 3])
        } else {
            panic!("Invalid shape")
        }
    }

    fn get_embed_positions(&mut self, input: &Tensor) -> Tensor {
        if self.embed_positions.device().ne(&input.device()) {
            self.embed_positions = self.embed_positions.to_device(input.device());
        }

        self.embed_positions.repeat([input.size()[0], 1, 1])
    }

    fn apply_rotary_pos_emb(tensor: &Tensor, sin: &Tensor, cos: &Tensor) -> Tensor {
        let repeats = 2;
        let dim = 3;

        let sin = sin
            .unsqueeze(dim - 1)
            .repeat_interleave_self_int(repeats, dim, None);
        let cos = cos
            .unsqueeze(dim - 1)
            .repeat_interleave_self_int(repeats, dim, None);

        let rotated = Self::rotate_every_two(tensor);

        tensor.multiply(&cos) + rotated.multiply(&sin)
    }

    fn rotate_every_two(tensor: &Tensor) -> Tensor {
        let rotate_dim = 3i64;
        let dim_count = tensor.size()[rotate_dim as usize];

        let zero_start_indices =
            Tensor::arange_start_step(0, dim_count, 2, (Kind::Int64, tensor.device()));
        let one_start_indices =
            Tensor::arange_start_step(1, dim_count, 2, (Kind::Int64, tensor.device()));

        let x_zero = tensor.index_select(rotate_dim, &zero_start_indices);
        let x_one = tensor.index_select(rotate_dim, &one_start_indices);

        let x = Tensor::stack(&[x_one.neg(), x_zero], -1);

        x.flatten(-2, -1)
    }
}

impl MLP {
    pub fn new(vb: &nn::Path, inter_size: usize, embed_size: usize, resid_pdrop: f32) -> MLP {
        let fc_in = linear(
            vb / "fc_in",
            embed_size as i64,
            inter_size as i64,
            LinearConfig {
                ..Default::default()
            },
        );

        let fc_out = linear(
            vb / "fc_out",
            inter_size as i64,
            embed_size as i64,
            LinearConfig {
                ..Default::default()
            },
        );

        MLP {
            fc_in,
            fc_out,
            resid_pdrop,
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut input = self.fc_in.forward(&input);
        let _ = input.gelu_("none");
        let mut input = self.fc_out.forward(&input);
        let _ = input.dropout_(self.resid_pdrop as f64, false);

        Ok(input)
    }
}

fn repeat_interleave(tensor: &Tensor, repeats: usize, dim: usize) -> Tensor {
    let dim = dim as i64;
    let tensor = tensor.unsqueeze(dim);

    let mut shape = tensor.size();
    shape[dim as usize] = repeats as i64;

    let mut tensor = tensor.broadcast_to(shape);
    let _ = tensor.transpose_(dim, dim + 1);
    let tensor = tensor.flatten(dim, dim + 1);
    let tensor = tensor.unsqueeze(dim);

    tensor
}
