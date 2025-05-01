import torch.nn as nn
import torch.quantization
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model = model.cuda()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer