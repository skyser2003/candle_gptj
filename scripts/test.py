from argparse import ArgumentParser

from transformers import GPTJModel, GPTJForCausalLM, AutoTokenizer, GenerationConfig
import torch


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Directory containing the model to test"
    )

    args = parser.parse_args()

    model_dir: str = args.model_dir

    model = GPTJForCausalLM.from_pretrained(model_dir, use_safetensors=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    inputs = ["Hello who are you?"]
    input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt")["input_ids"]

    result = model.forward(input_ids)
    logits = result["logits"]
    output_tokens = torch.argmax(logits, dim=-1)

    gen_config: GenerationConfig = GenerationConfig(
        num_beams=1,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_tokens = model.generate(input_ids, gen_config)

    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    print(outputs)


if __name__ == "__main__":
    main()
