use std::fs::File;

use candle_core::Device;
use memmap2::MmapOptions;
use safetensors::SafeTensors;

pub struct ModelLoader {}

impl ModelLoader {
    pub fn new(model_dir: &str, tokenizer_dir: &str, device: &Device) {
        println!("Loading model...");

        let model_dir = std::path::Path::new(model_dir);

        let filename = model_dir.join("model.safetensors");
        let file = File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();
        println!("{:?}", tensors.names());

        // let tensor = tensors.tensor("test").unwrap();

        println!("Loading model done")
    }
}
