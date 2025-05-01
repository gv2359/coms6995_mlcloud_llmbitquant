import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch.nn.functional as F

bitnet_storage_bytes = 0  # Global counter for packed weights

class BitPackedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Store 1-bit weights as packed bits
        packed_len = (in_features + 7) // 8
        self.register_buffer("packed_weight", torch.zeros(out_features, packed_len, dtype=torch.uint8))
        global bitnet_storage_bytes
        bitnet_storage_bytes += out_features * packed_len  # track memory usage

    def forward(self, input):
        # Unpack weights to binary {-1, +1}
        unpacked_weight = self.unpack(self.packed_weight)
        # Convert {0,1} to {-1,+1}
        binary_weight = unpacked_weight * 2 - 1
        return F.linear(input, binary_weight, self.bias)

    def pack(self, weight):
        # Convert from float to {0, 1} representation
        binary = (weight > 0).to(torch.uint8)
        packed = torch.zeros(weight.shape[0], (weight.shape[1] + 7) // 8, dtype=torch.uint8)
        for i in range(8):
            idx = (weight.shape[1] - 1 - i)
            if idx >= 0:
                packed |= (binary[:, idx::8] << i)
        return packed

    def unpack(self, packed):
        unpacked = torch.zeros(packed.size(0), packed.size(1) * 8, dtype=torch.float32, device=packed.device)
        for i in range(8):
            unpacked[:, i::8] = ((packed >> i) & 1).float()
        return unpacked[:, :self.in_features]


def apply_bitnet(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            for attr in name.split(".")[:-1]:
                parent = getattr(parent, attr)
            layer_name = name.split(".")[-1]

            old_layer = getattr(parent, layer_name)
            bit_layer = BitPackedLinear(old_layer.in_features, old_layer.out_features, bias=(old_layer.bias is not None))

            # Pack the weights
            with torch.no_grad():
                bit_layer.packed_weight.copy_(bit_layer.pack(old_layer.weight))
                if old_layer.bias is not None:
                    bit_layer.bias.data.copy_(old_layer.bias)

            setattr(parent, layer_name, bit_layer)
    return model


def load_model(model_name, device):
    global bitnet_storage_bytes
    bitnet_storage_bytes = 0  # reset before counting

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = apply_bitnet(model)
    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"BitNet packed weight size: {bitnet_storage_bytes / 1024:.2f} KB")

    return model, tokenizer