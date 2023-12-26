mod component;
mod test;

use anyhow::anyhow;
use clap::Parser;

use candle_core::{backend::BackendDevice, CudaDevice, Device};

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
async fn main() -> anyhow::Result<()> {
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
    println!("");

    let mut loader = ModelLoader::new(&model_dir, &tokenizer_dir, &device);
    let inputs = ["Hello who are you?"];
    let outputs = loader.inference(&inputs)?;

    println!("Inputs: {:?}", inputs);
    println!("Outputs: {:?}", outputs);

    Ok(())
}
