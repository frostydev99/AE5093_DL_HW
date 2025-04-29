# === train.py ===
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MagneticPINN import MagneticPINN_BField, pdeLoss_BField
from generateTrainingData import generateTrainingData

# === Utility ===
def normalize_tensor(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(0, keepdim=True)
    if std is None:
        std = x.std(0, keepdim=True)
    return (x - mean) / (std + 1e-8), mean, std

# === Seed & Device ===
seed = 45
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Hyperparameters ===
numSamples = 10000
numEpochs = 5000

# === Prepare Data
ecef_tensor, B_tensor, r_dipole, m_dipole, anchor_idx = generateTrainingData(numSamples)

# Normalize inputs and outputs
ecef_tensor, ecef_mean, ecef_std = normalize_tensor(ecef_tensor)
B_tensor, B_mean, B_std = normalize_tensor(B_tensor)

X = ecef_tensor
Y = B_tensor

# Pass to GPU
X = X.to(device)
Y = Y.to(device)
m_dipole = m_dipole.to(device)
r_dipole = r_dipole.to(device)
ecef_mean = ecef_mean.to(device)
ecef_std = ecef_std.to(device)

# === Setup Model ===
model = MagneticPINN_BField().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# === Train ===
totalLossHist = []
dataLossHist = []
pdeLossHist = []

for epoch in range(numEpochs):
    optim.zero_grad()

    Y_pred = model(X)
    loss_data = nn.functional.mse_loss(Y_pred, Y)
    loss_pde = pdeLoss_BField(model, X, ecef_mean, ecef_std)
    loss = loss_data + 1.0 * loss_pde

    loss.backward()
    optim.step()

    totalLossHist.append(loss.item())
    dataLossHist.append(loss_data.item())
    pdeLossHist.append(loss_pde.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{numEpochs} - Total Loss: {loss.item():.4f} - Data Loss: {loss_data.item():.4f} - PDE Loss: {loss_pde.item():.4f}")

# Save model and normalization
torch.save({
    'model_state_dict': model.state_dict(),
    'ecef_mean': ecef_mean,
    'ecef_std': ecef_std,
    'B_mean': B_mean,
    'B_std': B_std,
    'r_dipole': r_dipole,
}, 'trained_model.pth')

# === Training Plots ===
plt.figure(figsize=(12, 6))
plt.plot(totalLossHist, label='Total Loss', color='blue')
plt.plot(dataLossHist, label='Data Loss', color='orange')
plt.plot(pdeLossHist, label='PDE Loss', color='green')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Train Localizer ===
