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
