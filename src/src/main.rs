mod component;

use anyhow::anyhow;
use clap::Parser;

use candle_core::{backend::BackendDevice, CudaDevice, Device, Tensor};
use candle_nn::Dropout;

use component::model::ModelLoader;

#[derive(Parser, Debug)]
struct Arguments {
    #[arg(short, long)]
    model_dir: String,

    #[arg(short, long)]
    tokenizer_dir: String,

    #[arg(short, long)]
    device: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Arguments::parse();

    println!("args: {:?}", args);

    let model_dir = args.model_dir;
    let tokenizer_dir = args.tokenizer_dir;
    let device_str = args.device.clone();

    let device_result = || -> anyhow::Result<Device> {
        let device_str = device_str.unwrap_or("cpu".to_string());
        let cuda_id = device_str.split("cuda:").collect::<Vec<_>>();

        if cuda_id.len() != 2 {
            return Err(anyhow!(
                "Invalid cuda device format, should be in form of 'cuda:<integer>'"
            ));
        }

        let cuda_id = cuda_id[1].parse::<usize>()?;

        Ok(Device::Cuda(CudaDevice::new(cuda_id)?))
    }();

    let device = device_result.unwrap_or(Device::Cpu);

    if device.is_cpu() {
        println!("Using device 'cpu'");
    } else if device.is_cuda() {
        println!("Using device '{}'", args.device.unwrap());
    }

    let loader = ModelLoader::new(&model_dir, &tokenizer_dir, &device);

    println!("Tensor keys: {:?}", loader.get_tensors().names());
    println!(
        "Vocab size: {:?}",
        loader.get_tokenizer().get_vocab_size(true)
    );

    // Test
    let tensor1 = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();
    let tensor2 = Tensor::new(&[[3u32], [2], [1]], &device).unwrap();

    let result = tensor1.broadcast_mul(&tensor2);
    println!("{}", result.unwrap());

    let dropout = Dropout::new(0.5);
    let result = dropout.forward(&tensor1, true);

    println!("{:?}", result.unwrap());
}
