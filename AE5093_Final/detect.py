import torch
from MagneticPINN import MagneticPINN
from utils.coordinateTransforms import lla_to_ecef, normalize_positions
from utils.plotting import plot_disturbances
import pandas as pd
import numpy as np

# === Seed & Device ===
seed = 45
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Load Model ===
checkpoint = torch.load('trained_model.pth', map_location=device)
mean = checkpoint['mean']
std = checkpoint['std']

model = MagneticPINN().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

data = pd.read_csv('./data/sensor_data.csv')

lat = 42.2745
lon = -71.8063
alt = 180.0
n_samples = len(data)

latitudes = np.full((n_samples,), lat)
longitudes = np.full((n_samples,), lon)
altitudes = np.full((n_samples,), alt)

ecef_positions = lla_to_ecef(latitudes, longitudes, altitudes)
coords_norm = (ecef_positions - mean) / std
coords_tensor = torch.tensor(coords_norm, dtype=torch.float32, device=device)

B_measured = torch.tensor(data[['rawMagX', 'rawMagY', 'rawMagZ']].values, dtype=torch.float32, device=device)

# === Predict ideal fields and compute disturbance
with torch.no_grad():
    predicted_B = model(coords_tensor)
    delta_B = torch.norm(predicted_B - B_measured, dim=1).cpu().numpy()

plot_disturbances(latitudes, longitudes, delta_B)
