import time
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def log_gpu_memory():
    return {
        "allocated_MB": torch.cuda.memory_allocated() / 1024**2,
        "reserved_MB": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_MB": torch.cuda.max_memory_allocated() / 1024**2
    }

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "percent": 100 * trainable / total}

def save_metrics(metrics, experiment_id):
    os.makedirs(f"{metrics['output_dir']}", exist_ok=True)
    with open(f"{metrics['output_dir']}/{experiment_id}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def load_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

