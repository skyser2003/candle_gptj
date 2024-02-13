use std::collections::HashMap;
use std::time::SystemTime;

use axum::extract::Path;
use axum::http::StatusCode;
use clap::builder::Str;
use itertools::izip;
use tch::Device;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::{sleep, Duration};

use axum::extract::Json;
use axum::Extension;
use axum::{routing::post, Router};
use axum_macros;
use serde::{Deserialize, Serialize};

use crate::component::model_tch;
use crate::component::model_tch::GenerationConfig;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub n: Option<i64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum GenerateResponse {
    Ok(GenerateOkResult),
    Fail(GenerateFailResult),
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerateOkResult {
    pub text: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerateFailResult {
    pub texts: Vec<String>,
    pub message: String,
}

pub struct Server {
    router: Router,
    port: i32,
}

impl Server {
    pub fn new(
        port: i32,
        model_dir: &str,
        tokenizer_dir: &str,
        dtype: Option<String>,
        device: Device,
    ) -> Self {
        let batch_size_str = std::env::var("BATCH_SIZE");

        let batch_size = match batch_size_str {
            Ok(batch_size_str) => batch_size_str.parse().unwrap_or(10) as usize,
            Err(_) => 10,
        };

        println!("Batch size: {}", batch_size);

        let (tx, mut rx) = mpsc::channel::<MessageQueue>(batch_size * 5);
        let mut storage = MessageStorage::new(model_dir, tokenizer_dir, dtype, device);

        tokio::spawn(async move {
            loop {
                let recv = rx.recv().await;

                if let Some(mq) = recv {
                    storage.save(mq);

                    while let Ok(more_mq) = rx.try_recv() {
                        storage.save(more_mq);

                        if batch_size <= storage.len() {
                            break;
                        }
                    }
                }

                storage.process().await;

                sleep(Duration::from_nanos(1)).await;
            }
        });

        let router = Router::new()
            .route("/generate", post(generate))
            .layer(Extension(tx));

        Self { router, port }
    }

    pub async fn serve(&self) {
        let bind_addr = format!("0.0.0.0:{}", self.port);
        println!("Start server at http://{}", bind_addr);

        let listener = tokio::net::TcpListener::bind(bind_addr).await.unwrap();
        axum::serve(listener, self.router.clone()).await.unwrap();
    }
}
pub struct MessageStorage {
    model: model_tch::ModelLoader,
    message_count: usize,
    all_messages: Vec<MessageQueue>,
    begin_time: SystemTime,
}

pub type InferRequest = GenerateRequest;

#[derive(Clone, Debug)]
pub struct InferResponse {
    pub text: String,
}

#[derive(Debug)]
pub struct MessageQueue {
    message: InferRequest,
    tx: oneshot::Sender<Vec<InferResponse>>,
}

impl MessageStorage {
    fn new(model_dir: &str, tokenizer_dir: &str, dtype: Option<String>, device: Device) -> Self {
        let model = model_tch::ModelLoader::new(model_dir, tokenizer_dir, false, dtype, &device);

        MessageStorage {
            model,
            all_messages: vec![],
            message_count: Default::default(),
            begin_time: SystemTime::now(),
        }
    }

    fn save(&mut self, mq: MessageQueue) {
        self.message_count += 1;

        // println!("Messages count: {} {}", messages_count, self.message_count);

        self.all_messages.push(mq);
    }

    fn len(&self) -> usize {
        self.message_count
    }

    fn reset_state(&mut self) {
        self.message_count = 0;
        self.all_messages.clear();

        self.begin_time = SystemTime::now();
    }

    async fn process(&mut self) {
        let mut all_messages = vec![];

        all_messages.append(&mut self.all_messages);

        self.reset_state();

        let reqs = all_messages
            .iter()
            .map(|mq| &mq.message)
            .collect::<Vec<_>>();

        let all_results = self.get_inference_result(&reqs).await;

        let mut all_results_iter = all_results.into_iter();

        for mq in all_messages {
            // TODO: currently sending single message
            let mut mq_results = vec![];

            let result = all_results_iter.next().unwrap();
            mq_results.push(result);

            match mq.tx.send(mq_results) {
                Ok(_) => {}
                Err(_) => {}
            }
        }
    }

    async fn get_inference_result(&mut self, messages: &Vec<&InferRequest>) -> Vec<InferResponse> {
        let messages_count = messages.len();

        if messages_count == 0 {
            return Vec::new();
        }

        println!("Messages count: {}", messages_count);

        let messages = messages.clone();

        // TODO: spawn blocking unncessary?
        let inputs = &messages
            .iter()
            .map(|msg| msg.prompt.as_str())
            .collect::<Vec<_>>();
        let configs = &messages
            .iter()
            .map(|msg| GenerationConfig {
                top_k: msg.top_k,
                top_p: msg.top_p,
                max_tokens: None,
                max_gen_tokens: msg.max_tokens,
            })
            .collect::<Vec<_>>();

        // TODO: configs per input
        let output_res = self.model.inference(&inputs, Some(configs[0].clone()));

        let res = if let Ok(outputs) = output_res {
            outputs
                .into_iter()
                .map(|text| InferResponse { text })
                .collect()
        } else {
            inputs
                .iter()
                .map(|_| InferResponse {
                    text: "".to_string(),
                })
                .collect()
        };

        res
    }
}

#[axum_macros::debug_handler]
async fn generate(
    Extension(storage_tx): Extension<mpsc::Sender<MessageQueue>>,
    Json(payload): Json<GenerateRequest>,
) -> (StatusCode, Json<GenerateResponse>) {
    let (tx, rx) = oneshot::channel();

    let mq = MessageQueue {
        message: payload,
        tx,
    };

    match storage_tx.send(mq).await {
        Ok(_) => {}
        Err(_) => {}
    };

    let llm_results = rx.await.unwrap();

    let result = GenerateOkResult {
        text: llm_results.iter().map(|res| res.text.clone()).collect(),
    };

    (StatusCode::OK, Json(GenerateResponse::Ok(result)))
}
