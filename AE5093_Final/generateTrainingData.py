# === generateTrainingData.py ===
import torch
import numpy as np
from wmm2020 import wmm

# === Magnetic field from WMM ===
def get_WMM_field(lat, lon, alt):
    field = wmm(lat, lon, alt / 1000.0, 2025)
    return np.array([
        field['north'].data,
        field['east'].data,
        field['down'].data
    ], dtype=np.float32).reshape(3)

# === LLA to ECEF ===
def lla_to_ecef(lat, lon, alt):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.stack((X, Y, Z), axis=-1)

# === Dipole magnetic disturbance ===
def errorSource(r_sensor, r_dipole, m_dipole):
    mu_0 = 4 * np.pi * 1e-7
    r = r_sensor - r_dipole
    r_norm = torch.norm(r, dim=1, keepdim=True) + 1e-8
    r_hat = r / r_norm
    if m_dipole.shape[0] == 1:
        m_dipole = m_dipole.expand(r.shape[0], -1)
    m_dot_r = torch.sum(m_dipole * r, dim=1, keepdim=True)
    B = (mu_0 / (4 * np.pi)) * (3 * r_hat * m_dot_r - m_dipole) / (r_norm**3)
    return B

# === Data generation ===
def generateTrainingData(n_samples=5000):
    lat = np.random.uniform(-60, 60, n_samples)
    lon = np.random.uniform(-180, 180, n_samples)
    alt = np.random.uniform(0, 1000, n_samples)
    lla = np.stack([lat, lon, alt], axis=1)
    lla_tensor = torch.tensor(lla, dtype=torch.float32)

    ecef = lla_to_ecef(lat, lon, alt)
    ecef_tensor = torch.tensor(ecef, dtype=torch.float32)

    B_array = np.array([get_WMM_field(lat[i], lon[i], alt[i]) for i in range(n_samples)])
    B_background = torch.tensor(B_array, dtype=torch.float32)

    anchor_idx = np.random.randint(n_samples)
    anchor_ecef = ecef_tensor[anchor_idx].unsqueeze(0)

    r_dipole = anchor_ecef + torch.tensor(np.random.uniform(-2, 2, (1, 3)), dtype=torch.float32)
    m_dipole = torch.tensor(np.random.uniform(-5, 5, (1, 3)), dtype=torch.float32)

    B_disturbance = errorSource(ecef_tensor, r_dipole, m_dipole)
    B_total = B_background + B_disturbance

    return ecef_tensor, B_total, r_dipole, m_dipole, anchor_idx