import torch
import torch.nn as nn

import arch

model = arch.model()
state_dict = torch.load('model.pth')

model.load_state_dict(state_dict)

x = torch.Tensor([[float(10)]])
x_norm = (x - 95.3061) / 49.6107 # Using raw training values

with torch.no_grad():
    output = model(x_norm)
    out = output.item() * 161561.8906 + 324908.1562
print(out)