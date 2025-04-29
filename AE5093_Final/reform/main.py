# === main.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

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
lr_mag = 0.0003
lr_loc = 0.0001

# === Prepare Data ===
llaTensor, ecefTensor, B_background, B_disturbance, B_tensor, r_dipole, m_dipole = generateTrainingData(numSamples)

# Normalize
ecefTensor, ecef_mean, ecef_std = normalizeTensor(ecefTensor)
B_tensor, B_mean, B_std = normalizeTensor(B_tensor)

# Split into train/test
train_idx, test_idx = train_test_split(np.arange(numSamples), test_size=0.2, random_state=seed)
ecef_train, ecef_test = ecefTensor[train_idx].to(device), ecefTensor[test_idx].to(device)
B_train, B_test = B_tensor[train_idx].to(device), B_tensor[test_idx].to(device)
r_dipole = r_dipole.to(device)
m_dipole = m_dipole.to(device)
ecef_mean = ecef_mean.to(device)
ecef_std = ecef_std.to(device)
B_mean = B_mean.to(device)
B_std = B_std.to(device)

# === Setup Model ===
magModel = MagneticPINN().to(device)
magOptim = torch.optim.Adam(magModel.parameters(), lr=lr_mag)

# === Train Magnetic Model ===
totalLossHist_mag = []
dataLossHist_mag = []
pdeLossHist_mag = []

for epoch in range(numEpochs):
    magOptim.zero_grad()
    Y_pred = magModel(ecef_train)
    lossData = F.mse_loss(Y_pred, B_train)
    lossPDE = magModel.pdeLoss(ecef_train, ecef_mean, ecef_std)
    loss = lossData + 1.0 * lossPDE
    loss.backward()
    magOptim.step()

    totalLossHist_mag.append(loss.item())
    dataLossHist_mag.append(lossData.item())
    pdeLossHist_mag.append(lossPDE.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Total Loss: {loss.item():.4f} | Data Loss: {lossData.item():.4f} | PDE Loss: {lossPDE.item():.4f}")

# === Mag Test Set Evaluation ===
magModel.eval()
with torch.no_grad():
    B_test_pred = magModel(ecef_test)
    test_loss_data = F.mse_loss(B_test_pred, B_test).item()
print(f"[TEST] MagneticPINN Data Loss: {test_loss_data:.6e}")

# === Save Magnetic Model ===
torch.save({
    'model_state_dict': magModel.state_dict(),
    'ecef_mean': ecef_mean,
    'ecef_std': ecef_std,
    'B_mean': B_mean,
    'B_std': B_std,
    'r_dipole': r_dipole,
    'm_dipole': m_dipole,
}, "magModel.pth")

# === Train Localizer ===
grid_N = 8
span = 2e6
x = torch.linspace(-span, span, grid_N)
y = torch.linspace(-span, span, grid_N)
z = torch.linspace(-span, span, grid_N)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(device)
coords_norm = (coords - ecef_mean) / (ecef_std + 1e-8)

with torch.no_grad():
    B_pred = magModel(coords_norm) * B_std + B_mean

X_input = B_pred.view(-1).unsqueeze(0)
Y_target = r_dipole.unsqueeze(0)

locModel = LocalizerNN(X_input.shape[1]).to(device)
locOptim = torch.optim.Adam(locModel.parameters(), lr=lr_loc)
lossFN = nn.MSELoss()

lossHist = []
numEpochs_loc = 3000

for epoch in range(numEpochs_loc):
    locOptim.zero_grad()
    pred = locModel(X_input)
    loss = lossFN(pred, Y_target)
    loss.backward()
    locOptim.step()
    lossHist.append(loss.item())

    if epoch % 100 == 0:
        err = torch.norm(pred - Y_target).item()
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.8e} | Position Error: {err:.8f} m")

# === Save Localizer Model ===
pred_dipole = locModel(X_input).detach().cpu().numpy().squeeze()
r_dipole_true = r_dipole.cpu().numpy().squeeze()

torch.save({
    'model_state_dict': locModel.state_dict(),
    'input_size': X_input.shape[1],
    'true_r_dipole': r_dipole_true,
}, "locModel.pth")

print("=== Final Results ===")
print(f"True Dipole Position: {r_dipole_true}")
print(f"Predicted Dipole Position: {pred_dipole}")
print(f"Error: {np.linalg.norm(pred_dipole - r_dipole_true):.2f} m")

plt.figure(figsize=(10, 5))
plt.plot(lossHist)
plt.title("Localization Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()

# === Test Set Localization ===
coords_test = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(device)
coords_test_norm = (coords_test - ecef_mean) / (ecef_std + 1e-8)

with torch.no_grad():
    B_test_grid = magModel(coords_test_norm) * B_std + B_mean
    X_input_test = B_test_grid.view(-1).unsqueeze(0)
    pred_test_dipole = locModel(X_input_test).cpu().numpy().squeeze()

print("\n=== Test Grid Localization ===")
print(f"Predicted Dipole: {pred_test_dipole}")
print(f"True Dipole: {r_dipole_true}")
print(f"Test Grid Error: {np.linalg.norm(pred_test_dipole - r_dipole_true):.2f} m")