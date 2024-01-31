use std::collections::HashMap;
use std::time::SystemTime;

use axum::extract::Path;
use axum::http::StatusCode;
use itertools::izip;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::{sleep, Duration};

use axum::extract::Json;
use axum::Extension;
use axum::{routing::post, Router};
use axum_macros;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TritonRequest {
    pub inputs: Vec<TritonRequestInput>,
    pub outputs: Option<Vec<TritonRequestOutput>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum TritonResponse {
    Ok(TritonOkResult),
    Fail(TritonFailResult),
}

#[derive(Debug, Clone, Serialize)]
pub struct TritonOkResult {
    pub model_name: String,
    pub model_version: String,
    pub outputs: Vec<TritonResultOutput>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TritonFailResult {
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TritonRequestInput {
    pub name: String,
    pub shape: Vec<i32>,
    pub datatype: String,
    pub data: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TritonRequestOutput {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TritonResultOutput {
    name: String,
    datatype: String,
    shape: Vec<usize>,
    data: Vec<TritonOutputType>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum TritonOutputType {
    BYTES(String),
    FP32(f64),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let batch_size_str = std::env::var("BATCH_SIZE");

    let batch_size = match batch_size_str {
        Ok(batch_size_str) => batch_size_str.parse().unwrap_or(512) as usize,
        Err(_) => 512,
    };

    println!("Batch size: {}", batch_size);

    // TODO initialize model

    let langs = ["ko"];
    let langs = langs.map(|lang| lang.to_string());

    let (tx, mut rx) = mpsc::channel::<(String, MessageQueue)>(batch_size * 5);
    let mut storage = MessageStorage::new(&langs);

    tokio::spawn(async move {
        loop {
            let recv = rx.recv().await;

            if let Some((lang, mq)) = recv {
                storage.save(&lang, mq);

                while let Ok((more_lang, more_mq)) = rx.try_recv() {
                    storage.save(&more_lang, more_mq);

                    if batch_size <= storage.len() {
                        break;
                    }
                }
            }

            storage.process().await;

            sleep(Duration::from_nanos(1)).await;
        }
    });

    let app = Router::new()
        .route("/infer/:lang", post(spam_filter))
        .layer(Extension(tx));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    let result = axum::serve(listener, app);

    println!("{:?}", result);
    Ok(())
}

pub struct MessageStorage {
    langs: Vec<String>,
    all_messages: HashMap<String, Vec<MessageQueue>>,
    message_count: usize,
    begin_time: SystemTime,
}

#[derive(Debug)]
pub struct MessageQueue {
    messages: Vec<String>,
    tx: oneshot::Sender<Vec<(String, f64)>>,
}

impl MessageStorage {
    fn new(langs: &[String]) -> Self {
        let all_messages = langs
            .iter()
            .map(|lang| (lang.clone(), Vec::new()))
            .collect::<HashMap<_, _>>();

        MessageStorage {
            langs: langs.to_vec(),
            message_count: Default::default(),
            all_messages,
            begin_time: SystemTime::now(),
        }
    }

    fn save(&mut self, lang: &String, mq: MessageQueue) {
        self.message_count += mq.messages.len();

        // println!("Messages count: {} {}", messages_count, self.message_count);

        self.all_messages.get_mut(lang).unwrap().push(mq);
    }

    fn len(&self) -> usize {
        self.message_count
    }

    fn reset_state(&mut self) {
        self.message_count = 0;

        for messages in self.all_messages.values_mut() {
            messages.clear();
        }

        self.begin_time = SystemTime::now();
    }

    fn serialize(all_messages: &Vec<MessageQueue>) -> Vec<String> {
        let mut outputs = vec![];

        for mq in all_messages {
            for msg in mq.messages.iter() {
                outputs.push(msg.clone());
            }
        }

        outputs
    }

    async fn process(&mut self) {
        let mut all_messages = HashMap::new();

        for lang in self.langs.iter() {
            all_messages.insert(
                lang.clone(),
                self.all_messages.insert(lang.clone(), Vec::new()).unwrap(),
            );
        }

        self.reset_state();

        let flattened = all_messages
            .iter()
            .map(|(lang, mqs)| (lang.clone(), Self::serialize(mqs)))
            .collect::<HashMap<_, _>>();

        let all_results = get_spam_result(&flattened).await;

        for (lang, results) in all_results.into_iter() {
            let mqs = all_messages.remove(&lang).unwrap();

            let mut results_iter = results.into_iter();

            for mq in mqs.into_iter() {
                let mut mq_results = vec![];

                for _ in 0..mq.messages.len() {
                    let result = results_iter.next().unwrap();
                    mq_results.push(result.clone());
                }

                match mq.tx.send(mq_results) {
                    Ok(_) => {}
                    Err(_) => {}
                }
            }
        }
    }
}

async fn get_spam_result(
    messages: &HashMap<String, Vec<String>>,
) -> HashMap<String, Vec<(String, f64)>> {
    let messages_count = messages.iter().map(|(_, msgs)| msgs.len()).sum::<usize>();

    if messages_count == 0 {
        return HashMap::new();
    }

    println!("Messages count: {}", messages_count);

    let messages = messages.clone();

    let res: HashMap<String, (Vec<String>, Vec<f64>)> = tokio::task::spawn_blocking(move || {
        // TODO: execute model
        HashMap::new()
    })
    .await
    .unwrap();

    res.iter()
        .map(|(lang, (res1, res2))| {
            (
                lang.clone(),
                izip!(res1, res2)
                    .map(|(val1, val2)| (val1.clone(), *val2))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<HashMap<_, _>>()
}

#[axum_macros::debug_handler]
async fn spam_filter(
    Path(lang): Path<String>,
    Extension(storage_tx): Extension<mpsc::Sender<(String, MessageQueue)>>,
    Json(payload): Json<TritonRequest>,
) -> (StatusCode, Json<TritonResponse>) {
    let messages = &payload.inputs[0].data;

    let (tx, rx) = oneshot::channel();

    let mq = MessageQueue {
        messages: messages.clone(),
        tx,
    };

    match storage_tx.send((lang, mq)).await {
        Ok(_) => {}
        Err(_) => {}
    };

    let py_results = rx.await.unwrap();

    let mut results = TritonOkResult {
        model_name: "spam_filter_svm".to_string(),
        model_version: "1".to_string(),
        outputs: vec![
            TritonResultOutput {
                name: "OUTPUT0".to_string(),
                datatype: "BYTES".to_string(),
                shape: vec![py_results.len()],
                data: vec![],
            },
            TritonResultOutput {
                name: "OUTPUT1".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![py_results.len()],
                data: vec![],
            },
        ],
    };

    for (val1, val2) in py_results {
        results.outputs[0].data.push(TritonOutputType::BYTES(val1));
        results.outputs[1].data.push(TritonOutputType::FP32(val2));
    }

    (StatusCode::OK, Json(TritonResponse::Ok(results)))
}
