use candle_core::{Device, Tensor};
use candle_nn::Dropout;

fn main() {
    println!("Hello, world!");

    let device = Device::Cpu;

    let tensor = Tensor::new(&[2u32, 2, 2], &device).unwrap();
    let dropout = Dropout::new(0.5);
    let result = dropout.forward(&tensor, true);

    println!("{:?}", result.unwrap());
}
