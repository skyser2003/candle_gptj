use std::{fs::File, path::PathBuf};

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

pub struct ModelLoader {
    model: ModelWrapper,
    tokenizer: Tokenizer,
}

pub struct ModelWrapper {
    pub buffer: memmap2::Mmap,
}

impl ModelLoader {
    pub fn new(model_dir: &str, tokenizer_dir: &str) -> ModelLoader {
        println!("Begin loading model...");

        let model_dir = std::path::Path::new(model_dir);
        let model_filename = model_dir.join("model.safetensors");
        let model = Self::load_model(&model_filename);

        let tokenizer_dir = std::path::Path::new(tokenizer_dir);
        let tokenizer_filename = tokenizer_dir.join("tokenizer.json");
        let tok = Tokenizer::from_file(tokenizer_filename).unwrap();

        println!("Loading model done");

        Self {
            model: model,
            tokenizer: tok,
        }
    }

    pub fn load_model(model_filename: &PathBuf) -> ModelWrapper {
        let file = File::open(model_filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };

        let wrapper = ModelWrapper { buffer: buffer };

        return wrapper;
    }

    pub fn get_tensors(&self) -> SafeTensors {
        self.model.tensors()
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

impl ModelWrapper {
    pub fn tensors(&self) -> SafeTensors {
        let tensors = SafeTensors::deserialize(&self.buffer).unwrap();

        return tensors;
    }
}
