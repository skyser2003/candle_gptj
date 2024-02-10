use std::time::Instant;

use anyhow::anyhow;
use candle_core::{backend::BackendDevice, CudaDevice, Device};
use criterion::{criterion_group, criterion_main, Criterion};

use candle_gptj::component::model_candle::ModelLoader;

fn bench(c: &mut Criterion) {
    c.bench_function("case1", |b| {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();

        let model_dir = manifest_dir.join("../model/gpt-j-6b");
        let model_dir = model_dir.to_str().unwrap();
        let tokenizer_dir = model_dir;
        let device_str = Some("cpu".to_string());

        let device_result = || -> anyhow::Result<Device> {
            let device_str = device_str.clone().unwrap_or("cpu".to_string());
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
            println!("Using device '{}'", device_str.unwrap());
        }
        println!("");

        let mut loader = ModelLoader::new(
            &model_dir,
            &tokenizer_dir,
            Some("float16".to_string()),
            &device,
        );
        let inputs = ["Hello who are you?", "What is your name?"];

        b.iter(|| {
            let start_time = Instant::now();
            let outputs = loader.inference(&inputs).unwrap();
            let end_time = Instant::now();

            println!("Inputs: {:?}", inputs);
            println!("Outputs: {:?}, length: {}", outputs, outputs.len());
            println!(
                "Total single token time: {}",
                (end_time - start_time).as_secs_f32()
            );
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
