import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalizerNN(nn.Module):
    def __init__(self, hiddenLayers, neuronsPerLayer, input_size=6):
        super(LocalizerNN, self).__init__()
        self.fc_input = nn.Linear(input_size, neuronsPerLayer)
        self.fc_hidden = nn.ModuleList([
            nn.Linear(neuronsPerLayer, neuronsPerLayer) for _ in range(hiddenLayers)
        ])
        self.fc_output = nn.Linear(neuronsPerLayer, 3)

    def forward(self, x):
        h = F.silu(self.fc_input(x))
        for layer in self.fc_hidden:
            h_prev = h
            h = F.silu(layer(h)) + h_prev
        return self.fc_output(h)