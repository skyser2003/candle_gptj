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

#[tokio::test]
async fn cpu_test() -> anyhow::Result<()> {
    use std::time::Instant;

    use candle_core::IndexOp;
    use candle_core::{Device, Tensor};

    // let device = Device::new_cuda(0).unwrap();
    let start = Instant::now();
    let device = Device::Cpu;

    let loop_count = 10000000;

    let tensor1 = Tensor::randn(0f32, 1f32, (loop_count, 3, 3), &device)?;
    let tensor2 = Tensor::randn(0f32, 1f32, (loop_count, 3, 3), &device)?;

    let mut res = Tensor::new(&[0u32], &device).unwrap();

    for i in 0..loop_count {
        let a = tensor1.i(i).unwrap();
        let b = tensor2.i(i).unwrap();

        res = a.matmul(&b).unwrap();
    }

    let end = start.elapsed().as_secs_f32();
    eprintln!("Elapsed time: {:?}s, {} times", end, loop_count);
    eprintln!("res = {}", res);

    Ok(())
}
