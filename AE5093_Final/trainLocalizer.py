import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from MagneticPINN import MagneticPINN_BField

# === CONFIG ===
grid_N = 8  # N x N x N sampling grid
samples_per_case = grid_N ** 3
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

# === Load trained PINN ===
checkpoint = torch.load("trained_model.pth", map_location=device)
model = MagneticPINN_BField().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

ecef_mean = checkpoint["ecef_mean"].to(device)
ecef_std = checkpoint["ecef_std"].to(device)
B_mean = checkpoint["B_mean"].to(device)
B_std = checkpoint["B_std"].to(device)
r_dipole_true = checkpoint["r_dipole"].cpu().numpy().squeeze()

# === Generate 3D sampling grid around origin ===
span = 2e6  # meters
x = torch.linspace(-span, span, grid_N)
y = torch.linspace(-span, span, grid_N)
z = torch.linspace(-span, span, grid_N)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(device)

# Normalize positions
coords_norm = (coords - ecef_mean) / (ecef_std + 1e-8)

# === Predict B field
with torch.no_grad():
    B_pred = model(coords_norm) * B_std + B_mean

B_flat = B_pred.view(-1)
X_input = B_flat.unsqueeze(0)  # shape (1, 3*N^3)
Y_target = torch.tensor(r_dipole_true, dtype=torch.float32).unsqueeze(0)  # shape (1, 3)

# === Define Localizer
class DipoleLocalizer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

localizer = DipoleLocalizer(X_input.shape[1]).to(device)
optimizer = torch.optim.Adam(localizer.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# === Train Localizer
epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = localizer(X_input.to(device))
    loss = loss_fn(pred, Y_target.to(device))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        err = torch.norm(pred - Y_target.to(device)).item()
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4e} | Position Error: {err:.2f} m")

# === Final Result
pred_dipole = localizer(X_input.to(device)).detach().cpu().numpy().squeeze()
print(f"\nPredicted dipole location: {pred_dipole}")
print(f"True dipole location     : {r_dipole_true}")
print(f"Localization error       : {np.linalg.norm(pred_dipole - r_dipole_true):.2f} meters")

# === Plot training curve
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Dipole Localizer Training Loss")
plt.grid()
plt.tight_layout()
plt.show()

