import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from MagneticPINN import MagneticPINN, totalLoss
from utils.coordinateTransforms import lla_to_ecef, normalize_positions

# === Seed & Device ===
seed = 45
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Load Data ===
data = pd.read_csv('./data/sensor_data.csv')

# Inject Coordinates
lat = 42.2745
lon = -71.8063
alt = 180.0
n_samples = len(data)

lat = np.full((n_samples,), lat)
lon = np.full((n_samples,), lon)
alt = np.full((n_samples,), alt)

# === Convert to ECEF ===
ecefPos = lla_to_ecef(lat, lon, alt)
coords_norm, mean, std = normalize_positions(ecefPos)

coordTensor = torch.tensor(coords_norm, dtype=torch.float32, device=device)

# Mag Data
B_measured = data[['rawMagX', 'rawMagY', 'rawMagZ']].values
B_measured = torch.tensor(B_measured, dtype=torch.float32, device=device)

# === Setup Model ===
model = MagneticPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10000):
    optimizer.zero_grad()
    loss = totalLoss(model, coordTensor)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save({'model_state_dict': model.state_dict(), 'mean': mean, 'std': std}, 'trained_model.pth')
