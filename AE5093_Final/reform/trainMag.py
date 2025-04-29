import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from models.MagneticPINN import MagneticPINN
from models.LocalizerNN import LocalizerNN

from utils.generateTrainingData import generateTrainingData

# === Normalize Tensor ===
def normalizeTensor(x, mean=None, std=None):
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

# === Prepare Data ===
ecefTensor, B_tensor, r_dipole, m_dipole, anchorIdx = generateTrainingData(numSamples)

# Normalize
ecefTensor, ecef_mean, ecef_std = normalizeTensor(ecefTensor)

B_tensor, B_mean, B_std = normalizeTensor(B_tensor)

# Pack data and pass to GPU
X = ecefTensor.to(device)
Y = B_tensor.to(device)
m_dipole = m_dipole.to(device)
r_dipole = r_dipole.to(device)
ecef_mean = ecef_mean.to(device)
ecef_std = ecef_std.to(device)

# === Setup Model ===
magModel = MagneticPINN().to(device)
localizerModel = LocalizerNN().to(device)

magOptim = torch.optim.Adam(magModel.parameters(), lr=0.001)
localizerOptim = torch.optim.Adam(localizerModel.parameters(), lr=0.001)

# === Train Magnetic Model ===
totalLossHist_mag = []
dataLossHist_mag = []
pdeLossHist_mag = []

for epoch in range(numEpochs):
    magOptim.zero_grad()

    Y_pred = magModel(X)

    lossData = F.mse_loss(Y_pred, Y)
    lossPDE = magModel.pdeLoss(X, ecef_mean, ecef_std)
    loss = lossData + 1.0 * lossPDE

    loss.backward()
    magOptim.step()

    totalLossHist_mag.append(loss.item())
    dataLossHist_mag.append(lossData.item())
    pdeLossHist_mag.append(lossPDE.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Total Loss: {loss.item():.4f}, Data Loss: {lossData.item():.4f}, PDE Loss: {lossPDE.item():.4f}")

torch.save({
    'model_state_dict': magModel.state_dict(),
    'ecef_mean': ecef_mean,
    'ecef_std': ecef_std,
    'B_mean': B_mean,
    'B_std': B_std,
    'r_dipole': r_dipole,
    'm_dipole': m_dipole,
}, "magModel.pth")