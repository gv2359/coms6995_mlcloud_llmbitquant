from bitnet_loader import BitLinearLayer
import torch

# Simulate bit-packed input (batch=2, in_features=16 => packed len = 2 bytes)
input_packed = torch.randint(0, 256, (2, 2), dtype=torch.uint8, device="cuda")
layer = BitLinearLayer(in_features=16, out_features=4).cuda()

# Dummy packed weights
layer.weight_packed.copy_(torch.randint(0, 256, (4, 2), dtype=torch.uint8))

# Run forward
output = layer(input_packed)
print(output)