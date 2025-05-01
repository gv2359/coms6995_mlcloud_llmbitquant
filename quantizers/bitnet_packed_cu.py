# File: bitpacked_gpt2.py

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from bitnet.bitnet_loader import BitLinearLayer


def pack_activations(x):
    # x: [batch_size, features], float tensor in {-1, +1}
    x = (x > 0).to(torch.uint8)  # Convert to 0 or 1
    batch_size, in_features = x.shape
    packed_len = (in_features + 7) // 8
    packed = torch.zeros((batch_size, packed_len), dtype=torch.uint8, device=x.device)
    for i in range(8):
        packed |= ((x[:, i::8] & 1) << i)
    return packed


class BitPackedGPT2(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = base_model.config
        self.embed = base_model.transformer.wte
        self.pos_embed = base_model.transformer.wpe
        self.drop = base_model.transformer.drop
        self.h = nn.ModuleList([self._convert_block(block) for block in base_model.transformer.h])
        self.ln_f = base_model.transformer.ln_f
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def _convert_block(self, block):
        # Replace nn.Linear layers in MLP and attention with BitLinearLayer
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                parent = block
                for attr in name.split(".")[:-1]:
                    parent = getattr(parent, attr)
                key = name.split(".")[-1]
                old_layer = getattr(parent, key)
                bit_layer = BitLinearLayer(old_layer.in_features, old_layer.out_features, bias=old_layer.bias is not None)
                with torch.no_grad():
                    packed_w = (old_layer.weight > 0).to(torch.uint8)
                    packed_len = (old_layer.in_features + 7) // 8
                    weight_packed = torch.zeros(old_layer.out_features, packed_len, dtype=torch.uint8)
                    for i in range(8):
                        weight_packed |= ((packed_w[:, i::8] & 1) << i)
                    bit_layer.weight_packed.copy_(weight_packed)
                    if old_layer.bias is not None:
                        bit_layer.bias.data.copy_(old_layer.bias.data)
                setattr(parent, key, bit_layer)
        return block

    def forward(self, input_ids):
        device = input_ids.device
        input_shape = input_ids.size()
        input_embeds = self.embed(input_ids)
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        hidden_states = input_embeds + self.pos_embed(position_ids)
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            # Simulated input packing (not yet 1-bit for activations)
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


def load_bitpacked_gpt2(model_name="gpt2"):
    model = BitPackedGPT2(model_name=model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model.cuda(), tokenizer
