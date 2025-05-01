# File: benchmark_inference.py

import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from bitnet.bitpacked_gpt2 import load_bitpacked_gpt2


def benchmark_model(model, tokenizer, prompt, device="cuda", runs=10):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            out = model(input_ids)
            if isinstance(out, tuple):
                out = out[0]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(runs):
        with torch.no_grad():
            out = model(input_ids)
            if isinstance(out, tuple):
                out = out[0]
    torch.cuda.synchronize()

    end = time.time()
    avg_latency = (end - start) / runs
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return avg_latency, peak_memory


def compute_dataset_perplexity(model, tokenizer, dataset, device="cuda", max_samples=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for example in dataset.select(range(max_samples)):
        input_ids = tokenizer(example["func_code_string"], return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


if __name__ == "__main__":
    prompt = "def hello_world():\n    print('Hello, world!')"
    runs = 20

    # Load eval dataset
    print("\n📂 Loading evaluation dataset...")
    dataset = load_dataset("code_search_net", "python", split="validation[:100]")

    print("\n🔹 Baseline GPT-2")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_latency, base_memory = benchmark_model(base_model, base_tokenizer, prompt, runs=runs)
    base_ppl = compute_dataset_perplexity(base_model, base_tokenizer, dataset)
    print(f"Average Latency: {base_latency:.4f} sec")
    print(f"Peak GPU Memory: {base_memory:.2f} MB")
    print(f"Perplexity: {base_ppl:.2f}")

    print("\n🔹 BitPacked GPT-2")
    bit_model, bit_tokenizer = load_bitpacked_gpt2("gpt2", "../bitnet/bitnet_model/bitpacked_model.pt")
    bit_latency, bit_memory = benchmark_model(bit_model, bit_tokenizer, prompt, runs=runs)
    bit_ppl = compute_dataset_perplexity(bit_model, bit_tokenizer, dataset)
    print(f"Average Latency: {bit_latency:.4f} sec")
    print(f"Peak GPU Memory: {bit_memory:.2f} MB")
    print(f"Perplexity: {bit_ppl:.2f}")

    print("\n📊 Comparison")
    print(f"Speedup from BitPacked: {base_latency / bit_latency:.2f}×")
    print(f"Memory Reduction: {base_memory / bit_memory:.2f}×")
    print(f"Perplexity Delta: {bit_ppl - base_ppl:+.2f}")
