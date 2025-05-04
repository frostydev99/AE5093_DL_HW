import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from models.MagneticPINN import MagneticPINN
from models.LocalizerNN import LocalizerNN

# === Setup ===
seed = 45
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Hyperparameters ===
numEpochs = 10000
lr_loc = 0.008
numHidden = 5
numNeurons = 256

# === Load MagneticPINN Checkpoint ===
checkpoint = torch.load("magModel.pth", map_location=device)

magModel = MagneticPINN().to(device)
magModel.load_state_dict(checkpoint['model_state_dict'])
magModel.eval()

# === Load Data ===
ecef_train = checkpoint['r_sensor'].to(device)
B_train = checkpoint['B_train'].to(device)
r_dipole_train = checkpoint['r_dipole_train'].to(device)

# === Compute Residual Field ===
with torch.no_grad():
    B_background = magModel(ecef_train).detach()
    B_residual = B_train - B_background

print("Residual B-field magnitude (mean):", torch.norm(B_residual, dim=1).mean().item(), "μT")

# === Define Input and Output ===
X_input = torch.cat([B_residual, ecef_train], dim=1)               # (B, 6)
Y_target = r_dipole_train - ecef_train                             # Δr = r_dipole - r_sensor

# === Normalize Input and Output ===
X_mean = X_input.mean(0, keepdim=True)
X_std = X_input.std(0, keepdim=True)
X_input_norm = (X_input - X_mean) / (X_std + 1e-8)

Y_mean = Y_target.mean(0, keepdim=True)
Y_std = Y_target.std(0, keepdim=True)
Y_target_norm = (Y_target - Y_mean) / (Y_std + 1e-8)

# === Model, Optimizer, Scheduler ===
locModel = LocalizerNN(hiddenLayers=numHidden, neuronsPerLayer=numNeurons, input_size=6).to(device)
locOptim = torch.optim.Adam(locModel.parameters(), lr=lr_loc)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(locOptim, factor=0.5, patience=100, verbose=True)
lossFN = nn.MSELoss()

lossHist = []

best_loss = float('inf')
bestModelState = None

# === Training Loop ===
for epoch in range(numEpochs):
    locModel.train()
    locOptim.zero_grad()

    pred_norm = locModel(X_input_norm)
    loss = lossFN(pred_norm, Y_target_norm)
    loss.backward()
    locOptim.step()
    scheduler.step(loss.item())

    # Compute error in meters
    with torch.no_grad():
        pred = pred_norm * (Y_std + 1e-8) + Y_mean
        pred_global = pred + ecef_train
        err = torch.norm(pred_global - r_dipole_train, dim=1).mean().item()

    lossHist.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6e} | Mean Pos Error: {err:.2f} m | LR: {locOptim.param_groups[0]['lr']:.2e}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        bestModelState = {
            'model_state_dict': locModel.state_dict(),
            'input_size': 6,
            'X_mean': X_mean.cpu(),
            'X_std': X_std.cpu(),
            'Y_mean': Y_mean.cpu(),
            'Y_std': Y_std.cpu(),
        }

# === Final Results ===
print("\n=== Final Results ===")
print(f"Mean Localization Error: {err:.2f} meters")

# === Save Model ===
# torch.save({
#     'model_state_dict': locModel.state_dict(),
#     'input_size': 6,
#     'X_mean': X_mean.cpu(),
#     'X_std': X_std.cpu(),
#     'Y_mean': Y_mean.cpu(),
#     'Y_std': Y_std.cpu(),
# }, "locModel.pth")
torch.save(bestModelState, "locModel.pth")

# === Plot Loss ===
plt.figure(figsize=(10, 5))
plt.plot(lossHist)
plt.title("Localization Training Loss (Smooth L1)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()
