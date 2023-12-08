use candle_core::Device;

pub struct ModelLoader {}

impl ModelLoader {
    pub fn new(model_dir: &str, tokenizer_dir: &str, device: &Device) {
        println!("Loading model...");

        let model_dir = std::path::Path::new(model_dir);

        println!("Loading model done")
    }
}
