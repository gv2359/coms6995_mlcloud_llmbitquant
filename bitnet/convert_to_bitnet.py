# File: convert_to_bitpacked.py

import torch
from transformers import GPT2LMHeadModel
from bitnet_loader import BitLinearLayer
import os


def convert_linear_to_bitlinear(layer: torch.nn.Linear) -> BitLinearLayer:
    bit_layer = BitLinearLayer(layer.in_features, layer.out_features, bias=(layer.bias is not None))
    with torch.no_grad():
        # Convert float weights to 1-bit packed format
        binary = (layer.weight > 0).to(torch.uint8)
        packed_len = (layer.in_features + 7) // 8
        packed = torch.zeros(layer.out_features, packed_len, dtype=torch.uint8)
        for i in range(8):
            packed |= ((binary[:, i::8] & 1) << i)
        bit_layer.weight_packed.copy_(packed)
        if layer.bias is not None:
            bit_layer.bias.data.copy_(layer.bias.data)
    return bit_layer


def replace_linear_modules(model: torch.nn.Module):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            bit_layer = convert_linear_to_bitlinear(module)
            setattr(model, name, bit_layer)
        else:
            replace_linear_modules(module)


def convert_model_to_bitpacked(model_path: str, output_path: str):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    print("Loaded model from:", model_path)

    replace_linear_modules(model)
    print("Replaced nn.Linear with BitLinearLayer")

    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, "bitpacked_model.pt"))
    print("Saved bitpacked model weights to:", output_path)


if __name__ == "__main__":
    convert_model_to_bitpacked("../saved_models/baseline_gpt2", "./bitnet_model")
