mod component;
mod test;

use anyhow::anyhow;
use clap::Parser;

use candle_core::{backend::BackendDevice, CudaDevice};

use component::model_candle;
use component::model_tch;
use tokio::time::Instant;

#[derive(Parser, Debug)]
struct Arguments {
    #[arg(short, long)]
    model_dir: String,

    #[arg(short, long)]
    tokenizer_dir: String,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    device: Option<String>,

    #[arg(short, long)]
    framework: Option<String>,
}

enum FrameWorkType {
    Candle,
    Torch,
}

#[derive(PartialEq)]
enum DeviceType {
    Cpu,
    Cuda(usize),
}

fn get_candle_device(device: DeviceType) -> candle_core::Device {
    match device {
        DeviceType::Cpu => candle_core::Device::Cpu,
        DeviceType::Cuda(cuda_id) => {
            candle_core::Device::Cuda(candle_core::CudaDevice::new(cuda_id).unwrap())
        }
    }
}

fn get_tch_device(device: DeviceType) -> tch::Device {
    match device {
        DeviceType::Cpu => tch::Device::Cpu,
        DeviceType::Cuda(cuda_id) => tch::Device::Cuda(cuda_id),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Arguments::parse();

    println!("args: {:?}", args);

    let model_dir = args.model_dir;
    let tokenizer_dir = args.tokenizer_dir;
    let dtype_str = args.dtype.clone();
    let device_str = args.device.clone();
    let framework = args.framework.clone();

    let device_result = || -> anyhow::Result<DeviceType> {
        let device_str = device_str.unwrap_or("cpu".to_string());
        let cuda_id = device_str.split("cuda:").collect::<Vec<_>>();

        if cuda_id.len() != 2 {
            return Err(anyhow!(
                "Invalid cuda device format, should be in form of 'cuda:<integer>'"
            ));
        }

        let cuda_id = cuda_id[1].parse::<usize>()?;

        Ok(DeviceType::Cuda(cuda_id))
    }();

    let device_type = device_result.unwrap_or(DeviceType::Cpu);

    if device_type == DeviceType::Cpu {
        println!("Using device 'cpu'");
    } else {
        println!("Using device '{}'", args.device.unwrap());
    }
    println!("");

    let framework = if framework == Some("candle".to_string()) {
        FrameWorkType::Candle
    } else {
        FrameWorkType::Torch
    };

    let inputs = ["Hello who are you?", "What is your name?"];

    let (outputs, elapsed) = match framework {
        FrameWorkType::Candle => {
            let device = get_candle_device(device_type);
            let mut loader =
                model_candle::ModelLoader::new(&model_dir, &tokenizer_dir, dtype_str, &device);

            let start_time = Instant::now();
            let outputs = loader.inference(&inputs)?;
            let end_time = Instant::now();

            (outputs, end_time - start_time)
        }
        FrameWorkType::Torch => {
            let device = get_tch_device(device_type);
            let mut loader =
                model_tch::ModelLoader::new(&model_dir, &tokenizer_dir, dtype_str, &device);

            let start_time = Instant::now();
            let outputs = loader.inference(&inputs, None)?;
            let end_time = Instant::now();

            (outputs, end_time - start_time)
        }
        _ => unreachable!(),
    };

    println!("Inputs: {:?}", inputs);
    println!("Outputs: {:?}, length: {}", outputs, outputs.len());
    println!("Total single token time: {}", elapsed.as_secs_f32());

    Ok(())
}
