import torch
import torch.nn as nn

class LocalizerNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Output: estimated (x, y, z)
        )

    def forward(self, B_samples):
        return self.net(B_samples)
    
    