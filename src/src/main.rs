mod component;

use std::env;

use candle_core::{Device, Tensor};
use candle_nn::Dropout;

use component::model::ModelLoader;

#[tokio::main]
async fn main() {
    println!("Hello, world!");

    let model_dir = env::var("MODEL_DIR").unwrap();
    let tokenizer_dir = env::var("TOKENIZER_DIR").unwrap();

    let device = Device::Cpu;

    let loader = ModelLoader::new(&model_dir, &tokenizer_dir, &device);

    let tensor = Tensor::new(&[2u32, 2, 2], &device).unwrap();
    let dropout = Dropout::new(0.5);
    let result = dropout.forward(&tensor, true);

    println!("{:?}", result.unwrap());
}
