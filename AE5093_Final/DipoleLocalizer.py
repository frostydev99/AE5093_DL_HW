from torch import nn
import torch
import numpy as np

class DipoleLocalizer(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * num_samples, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output: estimated (x, y, z)
        )

    def forward(self, B_samples):
        return self.net(B_samples)
