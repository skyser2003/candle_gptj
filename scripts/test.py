from argparse import ArgumentParser
import time

from transformers import GPTJModel, GPTJForCausalLM, AutoTokenizer, GenerationConfig
import torch
import tqdm


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


def cpu_test():
    start_time = time.time()
    device = "cpu"
    loop_count = 10000000
    tensor_cpu = torch.randn((loop_count, 3, 3), device=device)
    tensor_cpu2 = torch.randn((loop_count, 3, 3), device=device)

    result_cpu = torch.tensor([0], device=device)

    # for a, b in tqdm.tqdm(zip(tensor_cpu, tensor_cpu2), total=tensor_cpu.shape[0]):
    for i in range(loop_count):
        a = tensor_cpu[i]
        b = tensor_cpu2[i]

        result_cpu = torch.mm(a, b)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed: {elapsed_time}s, {loop_count} times")
    print(f"res = {result_cpu}")


if __name__ == "__main__":
    # main()
    cpu_test()
