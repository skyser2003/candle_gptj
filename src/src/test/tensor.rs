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

#[cfg(test)]
const LOOP_COUNT: usize = 10;

#[cfg(test)]
const MAT_SIZE: usize = 3000;

#[cfg(test)]
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

#[tokio::test]
async fn candle_matmul_test() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;

    let repeat_count = 17;

    let t1 = Tensor::new(&[[1f32]], &device)?
        .repeat((repeat_count, repeat_count))?
        .unsqueeze(0)?;
    let t2 = Tensor::new(&[[1f32]], &device)?
        .repeat((repeat_count, repeat_count))?
        .unsqueeze(0)?;
    let target = Tensor::new(&[[1f32]], &device)?.repeat((repeat_count, repeat_count))?;

    let mul1 = t1.broadcast_matmul(&target)?;
    let mul2 = t2.broadcast_matmul(&target)?;

    println!("{}", mul1);
    println!("{}", mul2);

    Ok(())
}

#[tokio::test]
async fn candle_contiguous_test() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;

    let tensor = Tensor::new(&[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], &device)?;
    println!("{}", tensor.is_contiguous());
    assert_eq!(tensor.is_contiguous(), true);

    let tensor = tensor.unsqueeze(1)?;
    println!("{}", tensor.is_contiguous());
    assert_eq!(tensor.is_contiguous(), true);

    let tensor = tensor.repeat((2, 4))?;
    println!("{}", tensor.is_contiguous());
    assert_eq!(tensor.is_contiguous(), false);

    Ok(())
}
