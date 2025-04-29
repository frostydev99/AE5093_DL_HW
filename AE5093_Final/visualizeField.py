import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MagneticPINN import MagneticPINN_BField

# === Load model and normalization ===
checkpoint = torch.load("trained_model.pth", weights_only=True)
model = MagneticPINN_BField()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

ecef_mean = checkpoint["ecef_mean"]
ecef_std = checkpoint["ecef_std"]
B_mean = checkpoint["B_mean"]
B_std = checkpoint["B_std"]
r_dipole_true = checkpoint.get("r_dipole", None)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
model = model.to(device)

# === Generate 3D grid points near expected region
N = 10  # 10x10x10 = 1000 points
span = 2e6  # +- span in meters
x_vals = torch.linspace(-span, span, N)
y_vals = torch.linspace(-span, span, N)
z_vals = torch.linspace(-span, span, N)

X, Y, Z = torch.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (N^3, 3)

# Normalize coords
ecef_mean = ecef_mean.to(coords.device)
ecef_std = ecef_std.to(coords.device)
coords_norm = (coords - ecef_mean) / (ecef_std + 1e-8)

# Predict B field
with torch.no_grad():
    B_pred = model(coords_norm.to(device)) * B_std.to(device) + B_mean.to(device)

B_pred = B_pred.cpu().numpy()
coords = coords.cpu().numpy()

# === Estimate disturbance location
B_magnitude = np.linalg.norm(B_pred, axis=1)
predicted_idx = np.argmax(B_magnitude)
predicted_location = coords[predicted_idx]  # (3,)

# === Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot field vectors sparsely
skip = 5  # Plot every N-th vector
ax.quiver(coords[::skip, 0], coords[::skip, 1], coords[::skip, 2],
          B_pred[::skip, 0], B_pred[::skip, 1], B_pred[::skip, 2],
          length=2e5, normalize=True, color='gray', alpha=0.4)

# Plot predicted disturbance location
ax.scatter(predicted_location[0], predicted_location[1], predicted_location[2],
           c='blue', s=100, marker='o', label='Predicted Disturbance')

# Plot true disturbance location
if r_dipole_true is not None:
    r_dipole_true = r_dipole_true.cpu().numpy().squeeze()
    ax.scatter(r_dipole_true[0], r_dipole_true[1], r_dipole_true[2],
               c='red', s=100, marker='x', label='True Disturbance')

# Labels and view
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Magnetic Field and Disturbance Locations')
ax.legend()
ax.grid(True)
ax.set_box_aspect([1,1,1])  # Equal aspect ratio
plt.tight_layout()
plt.show()

# === Print locations
print(f"Predicted disturbance location (ECEF): {predicted_location}")
if r_dipole_true is not None:
    print(f"True disturbance location (ECEF): {r_dipole_true}")
    print(f"Error (meters): {np.linalg.norm(predicted_location - r_dipole_true):.2f} m")
