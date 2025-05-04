# === visualizeField.py ===

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.MagneticPINN import MagneticPINN
from models.LocalizerNN import LocalizerNN

# === Device Setup ===
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Load Magnetic Model ===
magCheckpoint = torch.load("magModel.pth", map_location=device)
magModel = MagneticPINN().to(device)
magModel.load_state_dict(magCheckpoint['model_state_dict'])
magModel.eval()

ecef_mean = magCheckpoint['ecef_mean']
ecef_std = magCheckpoint['ecef_std']
B_mean = magCheckpoint['B_mean']
B_std = magCheckpoint['B_std']

# === Load Localizer Model ===
locCheckpoint = torch.load("locModel.pth", map_location=device, weights_only=False)
input_size = locCheckpoint['input_size']
r_dipole_true = locCheckpoint['true_r_dipole']

locModel = LocalizerNN(input_size).to(device)
locModel.load_state_dict(locCheckpoint['model_state_dict'])
locModel.eval()

# === Generate Grid ===
grid_N = 8
span = 2e6  # meters
x = torch.linspace(-span, span, grid_N)
y = torch.linspace(-span, span, grid_N)
z = torch.linspace(-span, span, grid_N)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(device)

# Normalize ECEF positions
coords_norm = (coords - ecef_mean) / (ecef_std + 1e-8)

# === Predict B Field ===
with torch.no_grad():
    B_pred = magModel(coords_norm) * B_std + B_mean

# Flatten for localizer
B_flat = B_pred.view(-1)
X_input = B_flat.unsqueeze(0)  # (1, 3*N^3)

# === Predict Dipole Position ===
with torch.no_grad():
    pred_dipole = locModel(X_input).cpu().numpy().squeeze()

# === Print Results ===
print("=== Dipole Position Estimate ===")
print(f"True Dipole Position: {r_dipole_true}")
print(f"Predicted Dipole Position: {pred_dipole}")
print(f"Error: {np.linalg.norm(pred_dipole - r_dipole_true):.2f} m")

# === 3D Field Visualization ===
B_pred_np = B_pred.cpu().numpy()
coords_np = coords.cpu().numpy()

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot settings
skip = 5  # skip points to declutter arrows
scale = 2e5  # scale of arrows

# Normalize the field for uniform arrow size
B_mags = np.linalg.norm(B_pred_np, axis=1, keepdims=True)
B_dirs = B_pred_np / (B_mags + 1e-8)

ax.quiver(
    coords_np[::skip, 0],
    coords_np[::skip, 1],
    coords_np[::skip, 2],
    B_dirs[::skip, 0],
    B_dirs[::skip, 1],
    B_dirs[::skip, 2],
    length=scale,
    normalize=False,
    linewidth=0.5,
    arrow_length_ratio=0.3
)

# Plot true dipole
ax.scatter(r_dipole_true[0], r_dipole_true[1], r_dipole_true[2],
           color='green', s=120, label="True Dipole", marker='o')

# Plot predicted dipole
ax.scatter(pred_dipole[0], pred_dipole[1], pred_dipole[2],
           color='red', s=120, label="Predicted Dipole", marker='x')

# Axis labels
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Magnetic Field Vectors and Dipole Estimation', fontsize=16)

ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
