import os
from argparse import ArgumentParser

from transformers import AutoModelForCausalLM
from safetensors.torch import save_model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Directory containing the model to convert"
    )
    parser.add_argument(
        "--output_filename", help="Directory to save the converted model to"
    )

    args = parser.parse_args()

    model_dir: str = args.model_dir
    output_filename: str = args.output_filename

    if output_filename is None:
        output_filename = "./model.safetensors"

    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist")

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu")
    loaded = model.state_dict()

    loaded = {k: v.contiguous() for k, v in loaded.items()}

    save_model(model, output_filename)


if __name__ == "__main__":
    main()
