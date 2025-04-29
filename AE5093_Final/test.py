from generateTrainingData import generateTrainingData

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Generate data ===
n_samples = 1000
lla_tensor, B_tensor = generateTrainingData(n_samples)

# === Preprocess ===
lla = lla_tensor.cpu().numpy()  # (n_samples, 3)
B = B_tensor.cpu().numpy()      # (n_samples, 3)

# Correctly compute |B| (the field magnitude) — IMPORTANT
B_magnitude = np.linalg.norm(B, axis=1)  # (n_samples,)

# === Confirm shapes ===
print("lla.shape =", lla.shape)           # Should be (1000, 3)
print("B.shape =", B.shape)               # Should be (1000, 3)
print("B_magnitude.shape =", B_magnitude.shape)  # Should be (1000,)

# # === Plotting ===
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(16, 8))

# # 1. Scatter plot (longitude vs latitude) colored by magnetic field magnitude
# plt.subplot(1, 2, 1)
# scatter = plt.scatter(lla[:,1], lla[:,0], c=B_magnitude*1e6, cmap='viridis', s=20)  # µT
# plt.colorbar(scatter, label="|B| (µT)")
# plt.xlabel('Longitude (deg)')
# plt.ylabel('Latitude (deg)')
# plt.title('Magnetic Field Magnitude across Locations')

# # 2. Histogram of Magnetic Field Magnitude
# plt.subplot(1, 2, 2)
# plt.hist(B_magnitude*1e6, bins=50, color='navy', edgecolor='white')
# plt.xlabel('|B| (µT)')
# plt.ylabel('Count')
# plt.title('Distribution of Magnetic Field Magnitudes')

# plt.tight_layout()
# plt.show()
