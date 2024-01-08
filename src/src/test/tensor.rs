use tokio;

#[tokio::test]
async fn broadcast_mul() {
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;

    let tensor1 = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();
    let tensor2 = Tensor::new(&[[3u32], [2], [1]], &device).unwrap();

    let result = tensor1.broadcast_mul(&tensor2);
    println!("{}", result.unwrap());
}

#[tokio::test]
async fn dropout() {
    use candle_core::{Device, Tensor};
    use candle_nn::{self, Dropout};

    let device = Device::Cpu;

    let tensor1 = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();
    let dropout = Dropout::new(0.5);
    let result = dropout.forward(&tensor1, true);

    println!("{:?}", result.unwrap());
}

#[tokio::test]
async fn test_repeat_interleave() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};

    let dim = 1;
    let repeats = 3;
    let device = &Device::Cpu;

    let tensor = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], device).unwrap();
    let tensor = tensor.unsqueeze(1)?;

    let mut shape = tensor.dims().to_vec();
    shape[dim] = repeats;

    println!("{}\n", tensor);

    let tensor = tensor.broadcast_as(shape)?;
    println!("{}\n", tensor);

    let tensor = tensor.transpose(dim, dim + 1)?;
    println!("{}\n", tensor);

    let tensor = tensor.flatten(dim, dim + 1)?;
    println!("{}\n", tensor);

    let answer_tensor = Tensor::new(
        &[[1u32, 1, 1, 2, 2, 2, 3, 3, 3], [4, 4, 4, 5, 5, 5, 6, 6, 6]],
        device,
    )?;

    assert_eq!(tensor.to_vec2::<u32>()?, answer_tensor.to_vec2::<u32>()?);

    Ok(())
}

const LOOP_COUNT: usize = 10;
const MAT_SIZE: usize = 3000;

#[tokio::test]
async fn cpu_test_candle() -> anyhow::Result<()> {
    use std::hint::black_box;
    use std::time::Instant;

    use candle_core::IndexOp;
    use candle_core::{Device, Tensor};

    // let device = Device::new_cuda(0).unwrap();
    let device = Device::Cpu;

    let tensor1 = Tensor::randn(0f32, 1f32, (LOOP_COUNT, MAT_SIZE, MAT_SIZE), &device)?;
    let tensor2 = Tensor::randn(0f32, 1f32, (LOOP_COUNT, MAT_SIZE, MAT_SIZE), &device)?;

    let mut res = Tensor::new(&[0u32], &device).unwrap();

    let start = Instant::now();

    for i in 0..LOOP_COUNT {
        let a = tensor1.i(i)?;
        let b = tensor2.i(i)?;

        res = black_box(a.matmul(&b)?);
    }

    let end = start.elapsed().as_secs_f32();
    eprintln!("Elapsed time: {:?}s, {} times", end, LOOP_COUNT);
    eprintln!("res = {:?}", res.shape());

    Ok(())
}

async fn burn_test<B: burn::tensor::backend::Backend>(
    device: &B::Device,
    loop_count: usize,
    mat_size: usize,
) -> anyhow::Result<()> {
    use std::hint::black_box;
    use std::time::Instant;

    use burn::tensor;
    use burn::tensor::{Float, Int, Tensor};

    let tensor1 = Tensor::<B, 3>::random_device(
        [loop_count, mat_size, mat_size],
        tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let tensor2 = Tensor::<B, 3>::random_device(
        [loop_count, mat_size, mat_size],
        tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let mut res = Tensor::<B, 2, Float>::zeros([mat_size, mat_size]);

    let start = Instant::now();

    for i in 0..loop_count {
        let index_tensor = Tensor::<B, 1, Int>::from_ints([i as i32]);

        let a = tensor1.clone().select(0, index_tensor.clone()).squeeze(0);
        let b = tensor2.clone().select(0, index_tensor.clone()).squeeze(0);

        res = black_box(a.matmul(b));
    }

    let end = start.elapsed().as_secs_f32();
    eprintln!("Elapsed time: {:?}s, {} times", end, loop_count);
    eprintln!("res = {:?}", res.shape());

    Ok(())
}

#[tokio::test]
async fn cpu_test_burn_candle() -> anyhow::Result<()> {
    use burn::backend::{candle::CandleDevice, Candle};

    type CurrentBackend = Candle;
    let device = CandleDevice::Cpu;

    burn_test::<CurrentBackend>(&device, LOOP_COUNT, MAT_SIZE).await
}
#[tokio::test]
async fn cpu_test_burn_wgpu() -> anyhow::Result<()> {
    use burn::backend::{
        wgpu::{AutoGraphicsApi, WgpuDevice},
        Wgpu,
    };

    type CurrentBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    let device = WgpuDevice::Cpu;

    burn_test::<CurrentBackend>(&device, LOOP_COUNT, MAT_SIZE).await
}

#[tokio::test]
async fn cpu_test_burn_tch_cpu() -> anyhow::Result<()> {
    use burn::backend::{libtorch::LibTorchDevice, LibTorch};

    type CurrentBackend = LibTorch;
    let device = LibTorchDevice::default();

    burn_test::<CurrentBackend>(&device, LOOP_COUNT, MAT_SIZE).await
}

#[tokio::test]
async fn cpu_test_burn_tch_gpu() -> anyhow::Result<()> {
    use burn::backend::{libtorch::LibTorchDevice, LibTorch};

    type CurrentBackend = LibTorch;
    let device = LibTorchDevice::Cuda(0);

    burn_test::<CurrentBackend>(&device, LOOP_COUNT, MAT_SIZE).await
}
