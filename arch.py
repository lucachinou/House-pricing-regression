import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.sequential(x)