import torch
import torch.nn as nn

import arch

torch.manual_seed(42)

model = arch.model()
state_dict = torch.load('model.pth')

model.load_state_dict(state_dict)

x = torch.Tensor([[float(10)]])
x_norm = (x - 97.2449) / 51.1287 # Using raw training values

with torch.no_grad():
    output = model(x_norm)
    out = output.item() * 166949.4531 + 331408.1562
print(out)