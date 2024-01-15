from argparse import ArgumentParser
import time

from transformers import (
    GPTJModel,
    GPTJForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
)
import torch
import tqdm


def get_model(model_dir: str, device: str):
    print("Begin loading model...")
    start_time = time.time()

    model = GPTJForCausalLM.from_pretrained(model_dir, use_safetensors=False)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Hot loading
    hot_input_ids = tokenizer.batch_encode_plus(["Hot loading"], return_tensors="pt")[
        "input_ids"
    ]
    hot_input_ids = hot_input_ids.to(model.device)
    model.forward(hot_input_ids)

    end_time = time.time()

    print(f"Loading model done, {end_time - start_time}s")
    print()

    return model, tokenizer


def test_single(
    model: GPTJForCausalLM, tokenizer: PreTrainedTokenizer, inputs: list[str]
):
    start_time = time.time()

    input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)

    result = model.forward(input_ids)
    logits = result["logits"]
    output_tokens = torch.argmax(logits, dim=-1)

    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    end_time = time.time()

    print("Inputs: ", inputs)
    print("Outputs: ", outputs)
    print(f"Total single token time: {end_time - start_time}s")


def test_generate(
    model: GPTJForCausalLM, tokenizer: PreTrainedTokenizer, sentence: str
):
    inputs = [sentence]
    input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt")["input_ids"]

    gen_config: GenerationConfig = GenerationConfig(
        num_beams=1,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_tokens = model.generate(input_ids, gen_config)

    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    print(outputs)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Directory containing the model to test"
    )
    parser.add_argument("--device", required=False, default="cpu", help="Device")

    args = parser.parse_args()

    model_dir: str = args.model_dir
    device: str = args.device

    print(f"Using device '{device}'")
    print()

    model, tokenizer = get_model(model_dir, device)

    inputs: list[str] = ["Hello who are you?", "What is your name?"]
    test_single(model, tokenizer, inputs)
    # test_generate(model, tokenizer, inputs)


def cpu_test():
    device = "cpu"
    loop_count = 10
    mat_size = 3000
    tensor1 = torch.randn(
        (loop_count, mat_size, mat_size), dtype=torch.float32, device=device
    )
    tensor2 = torch.randn(
        (loop_count, mat_size, mat_size), dtype=torch.float32, device=device
    )

    result_cpu = torch.tensor([0], device=device)

    start_time = time.time()
    # for a, b in tqdm.tqdm(zip(tensor_cpu, tensor_cpu2), total=tensor_cpu.shape[0]):
    for i in range(loop_count):
        a = tensor1[i]
        b = tensor2[i]

        torch.mm(a, b)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed: {elapsed_time}s, {loop_count} times")
    print(f"res = {result_cpu.shape}")


if __name__ == "__main__":
    main()
    # cpu_test()
