import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Transformer
from matplotlib import cm


from models.MagneticPINN import MagneticPINN
from models.LocalizerNN import LocalizerNN

# === Device ===
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Load Magnetic Model ===
mag_ckpt = torch.load("magModel.pth", map_location=device)
magModel = MagneticPINN().to(device)
magModel.load_state_dict(mag_ckpt['model_state_dict'])
magModel.eval()

# === Load Localizer Model ===
loc_ckpt = torch.load("locModel.pth", map_location=device)
locModel = LocalizerNN(hiddenLayers=5, neuronsPerLayer=256, input_size=6).to(device)
locModel.load_state_dict(loc_ckpt['model_state_dict'])
locModel.eval()

# === Load data ===
r_sensor = mag_ckpt['r_sensor'].to(device)
r_dipole_true = mag_ckpt['r_dipole_train'].to(device)
B_train = mag_ckpt['B_train'].to(device)

X_mean = loc_ckpt['X_mean'].to(device)
X_std = loc_ckpt['X_std'].to(device)
Y_mean = loc_ckpt['Y_mean'].to(device)
Y_std = loc_ckpt['Y_std'].to(device)

# === Compute residual field and predictions ===
with torch.no_grad():
    B_background = magModel(r_sensor).detach()
    B_residual = B_train - B_background

    X_input = torch.cat([B_residual, r_sensor], dim=1)
    X_input_norm = (X_input - X_mean) / (X_std + 1e-8)
    pred_offset = locModel(X_input_norm)
    pred_offset = pred_offset * (Y_std + 1e-8) + Y_mean
    r_dipole_pred = pred_offset + r_sensor

# === Convert ECEF to LLA ===
transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

r_sensor_np = r_sensor.cpu().numpy()
r_true_np = r_dipole_true.cpu().numpy()
r_pred_np = r_dipole_pred.cpu().numpy()
B_np = B_residual.cpu().numpy()

N = 300
idx = np.random.choice(r_sensor_np.shape[0], N, replace=False)

# Sensor positions
lons, lats, alts = transformer.transform(r_sensor_np[idx, 0], r_sensor_np[idx, 1], r_sensor_np[idx, 2])

# True and predicted dipoles (same index)
true_lons, true_lats, true_alts = transformer.transform(
    r_true_np[idx, 0], r_true_np[idx, 1], r_true_np[idx, 2])
pred_lons, pred_lats, pred_alts = transformer.transform(
    r_pred_np[idx, 0], r_pred_np[idx, 1], r_pred_np[idx, 2])

# === Plot Setup ===
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot magnetic field vectors
ax.quiver(lons, lats, alts,
          B_np[idx, 0], B_np[idx, 1], B_np[idx, 2],
          length=40, normalize=True, color='grey', alpha=0.5, label='B_field')

# True & predicted dipole locations
ax.scatter(true_lons, true_lats, true_alts, c='green', s=10, label='True Dipole')
ax.scatter(pred_lons, pred_lats, pred_alts, c='red', s=10, label='Predicted Dipole')

# Error vectors
for i in range(len(idx)):
    ax.plot([true_lons[i], pred_lons[i]],
            [true_lats[i], pred_lats[i]],
            [true_alts[i], pred_alts[i]], color='gray', alpha=0.3)

# === Add Lat/Lon Grid ===
lat_grid = np.linspace(min(lats), max(lats), 15)
lon_grid = np.linspace(min(lons), max(lons), 15)
lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)

for i in range(lat_mesh.shape[0]):
    ax.plot(lon_mesh[i], lat_mesh[i], zs=0, color='lightgray', alpha=0.3)
for j in range(lat_mesh.shape[1]):
    ax.plot(lon_mesh[:, j], lat_mesh[:, j], zs=0, color='lightgray', alpha=0.3)

ax.set_title("Magnetic Field Vectors and Disturbance Locations")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.set_zlabel("Altitude [m]")
ax.legend()
plt.tight_layout()

# === Compute Euclidean Errors ===
errors = np.linalg.norm(r_pred_np[idx] - r_true_np[idx], axis=1)  # shape: (N,)
norm = plt.Normalize(errors.min(), errors.max())
colors = cm.viridis(norm(errors))  # Or cm.inferno, cm.plasma, etc.

# === Enhanced Error Visualization ===
fig_err = plt.figure(figsize=(12, 9))
ax_err = fig_err.add_subplot(111, projection='3d')

# Draw thick, colored lines from true → predicted
for i in range(len(idx)):
    ax_err.plot(
        [r_true_np[idx[i], 0], r_pred_np[idx[i], 0]],
        [r_true_np[idx[i], 1], r_pred_np[idx[i], 1]],
        [r_true_np[idx[i], 2], r_pred_np[idx[i], 2]],
        color=colors[i], linewidth=2.5, alpha=0.9
    )

# Plot points with larger sizes
ax_err.scatter(r_true_np[idx, 0], r_true_np[idx, 1], r_true_np[idx, 2],
               c=colors, s=50, label='True (color = error)', edgecolor='k')
ax_err.scatter(r_pred_np[idx, 0], r_pred_np[idx, 1], r_pred_np[idx, 2],
               c=colors, s=50, marker='^', label='Predicted', edgecolor='k')

# Add colorbar
mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
mappable.set_array(errors)
cbar = plt.colorbar(mappable, ax=ax_err, shrink=0.7)
cbar.set_label("Localization Error [m]")

ax_err.set_title("True vs. Predicted Disturbance Locations")
ax_err.set_xlabel("X [m]")
ax_err.set_ylabel("Y [m]")
ax_err.set_zlabel("Z [m]")
ax_err.legend()
plt.tight_layout()


plt.show()