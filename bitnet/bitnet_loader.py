# File: bitnet_loader.py

from torch.utils.cpp_extension import load
import os

# Compile and load the CUDA extension
bitnet_ops = load(
    name="bitnet_ops",
    sources=["../bitnet/bitnet_linear.cu"],
    verbose=True,
    extra_cuda_cflags=["--use_fast_math"]
)

# Python wrapper for the CUDA 1-bit linear operation
import torch
import torch.nn as nn

class BitLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_packed, weight_packed, in_features):
        batch_size = input_packed.size(0)
        out_features = weight_packed.size(0)
        output = torch.empty(batch_size, out_features, device=input_packed.device, dtype=torch.float32)
        bitnet_ops.bitlinear_forward(input_packed, weight_packed, output, in_features)
        return output


class BitLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        packed_len = (in_features + 7) // 8
        self.register_buffer("weight_packed", torch.zeros(out_features, packed_len, dtype=torch.uint8))

    def forward(self, input_packed):
        out = BitLinearFunction.apply(input_packed, self.weight_packed, self.in_features)
        if self.bias is not None:
            out += self.bias
        return out
