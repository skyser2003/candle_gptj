mod component;

use std::env;

use clap::Parser;

use candle_core::{Device, Tensor};
use candle_nn::Dropout;

use component::model::ModelLoader;

#[derive(Parser, Debug)]
struct Arguments {
    #[arg(short, long)]
    model_dir: String,

    #[arg(short, long)]
    tokenizer_dir: String,
}

#[tokio::main]
async fn main() {
    let args = Arguments::parse();

    println!("args: {:?}", args);

    let model_dir = args.model_dir;
    let tokenizer_dir = args.tokenizer_dir;

    let device = Device::Cpu;

    let loader = ModelLoader::new(&model_dir, &tokenizer_dir, &device);

    let tensor = Tensor::new(&[2u32, 2, 2], &device).unwrap();
    let dropout = Dropout::new(0.5);
    let result = dropout.forward(&tensor, true);

    println!("{:?}", result.unwrap());
}
