from argparse import ArgumentParser

from transformers import AutoModel, AutoTokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Directory containing the model to convert"
    )

    args = parser.parse_args()

    model_dir: str = args.model_dir

    model = AutoModel.from_pretrained(model_dir, device_map="cpu")
    print(model)


if __name__ == "__main__":
	main()