import json
from argparse import ArgumentParser
import time

from transformers import (
    GPTJModel,
    GPTJForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
    AutoConfig
)
import torch
import tqdm


dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat": torch.bfloat16,
}


def get_model(model_dir: str, dtype: str, device: str):
    print("Begin loading model...")
    start_time = time.time()

    if dtype == "":
        config = AutoConfig.from_pretrained(model_dir)

        if config.torch_dtype is not None:
            torch_dtype = config.torch_dtype
    else:
        torch_dtype = dtype_map[dtype]

    model = GPTJForCausalLM.from_pretrained(model_dir, use_safetensors=True, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Hot loading
    hot_input_ids = tokenizer.batch_encode_plus(["Hot loading"], return_tensors="pt")[
        "input_ids"
    ]
    hot_input_ids = hot_input_ids.to(model.device)
    gen_config = GenerationConfig(
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=1,
        top_p=1.0,
        max_length=5,
    )

    with torch.no_grad():
        model.generate(hot_input_ids, gen_config)

    end_time = time.time()

    print(f"Loading model done, dtype={model.dtype}, {end_time - start_time}s")
    print()

    return model, tokenizer


def test_single(
    model: GPTJForCausalLM, tokenizer: PreTrainedTokenizer, inputs: list[str]
):
    input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)

    result = model.forward(input_ids)
    logits = result["logits"]
    output_tokens = torch.argmax(logits, dim=-1)
    
    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    return outputs


def test_generate(
    model: GPTJForCausalLM, tokenizer: PreTrainedTokenizer, inputs: list[str]
):
    input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)

    gen_config = GenerationConfig(
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=1,
        top_p=1.0,
        max_length=50,
    )

    with torch.no_grad():
        output_tokens = model.generate(input_ids, gen_config)

    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Directory containing the model to test"
    )
    parser.add_argument("--dtype", required=False, default="", help="Dtype")
    parser.add_argument("--device", required=False, default="cpu", help="Device")

    args = parser.parse_args()

    model_dir: str = args.model_dir
    dtype: str = args.dtype
    device: str = args.device

    print(f"Using device '{device}'")
    print()

    model, tokenizer = get_model(model_dir, dtype, device)

    samples_file = open("../data/test.json", "r")
    samples = json.load(samples_file)
    samples_file.close()

    inputs: list[str] = samples["inputs"]
    start_time = time.time()
    
    # outputs = test_single(model, tokenizer, inputs)
    outputs = test_generate(model, tokenizer, inputs)

    end_time = time.time()

    print("Inputs: ", json.dumps(inputs, ensure_ascii=False))
    print("Outputs: ", json.dumps(outputs, ensure_ascii=False))
    print(f"Total single token time: {end_time - start_time}s")



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
