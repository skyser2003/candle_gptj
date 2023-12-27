from argparse import ArgumentParser

from transformers import GPTJModel, GPTJForCausalLM, AutoTokenizer
import torch


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Directory containing the model to test"
    )

    args = parser.parse_args()

    model_dir: str = args.model_dir

    model = GPTJModel.from_pretrained(model_dir, device_map="auto")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    inputs = ["Hello who are you?"]
    input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt")["input_ids"]

    result = model.forward(input_ids)

    hidden_state = result["last_hidden_state"]
    hidden_state = torch.argmax(hidden_state, dim=-1)

    outputs = tokenizer.batch_decode(hidden_state, skip_special_tokens=True)

    print(outputs)


if __name__ == "__main__":
    main()
