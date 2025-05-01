import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn

class BinarizeLinear(nn.Linear):
    def forward(self, input):
        binary_weight = torch.sign(self.weight)  # Approximate to -1, 0, +1
        return nn.functional.linear(input, binary_weight, self.bias)

def binarize_model(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            for attr in name.split(".")[:-1]:
                parent = getattr(parent, attr)
            layer_name = name.split(".")[-1]
            setattr(parent, layer_name, BinarizeLinear(module.in_features, module.out_features))
    return model


def load_model(model_name, device):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = binarize_model(model)
    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
