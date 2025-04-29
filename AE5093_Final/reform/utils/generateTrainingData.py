import torch
import numpy as np
from wmm2020 import wmm

from utils.coordinateTransforms import lla2ecef, ecef2lla

# === Magnetic field from WMM ===
def getWMMField(lat, lon, alt):
    B_wmm = wmm(lat, lon, alt / 1000.0, 2025)

    return np.array([
        B_wmm['north'].data,
        B_wmm['east'].data,
        B_wmm['down'].data
    ], dtype=np.float32).reshape(3)

# === Dipole Magnetic Disturbance ===
def errorSource(r_source, r_dipole, m_dipole):
    mu_0 = 4 * np.pi * 1e-7
    r = r_source - r_dipole
    r_norm = torch.norm(r, dim=1, keepdim=True) + 1e-8
    r_hat = r / r_norm

    if m_dipole.shape[0] == 1:
        m_dipole = m_dipole.expand(r.shape[0], -1)

    m_dot_r = torch.sum(m_dipole * r, dim=1, keepdim=True)
    B = (mu_0 / (4 * np.pi)) * (3 * r_hat * m_dot_r - m_dipole) / (r_norm**3)

    return B

# === Data Generation ===
def generateTrainingData(n_samples):
    lat = np.random.uniform(-60, 60, n_samples)
    lon = np.random.uniform(-180, 180, n_samples)
    alt = np.random.uniform(0, 1000, n_samples)
    lla = np.stack([lat, lon, alt], axis=1)
    lla_tensor = torch.tensor(lla, dtype=torch.float32)

    ecef = lla2ecef(lat, lon, alt)
    ecef_tensor = torch.tensor(ecef, dtype=torch.float32)

    B_array = np.array([getWMMField(lat[i], lon[i], alt[i]) for i in range(n_samples)])
    B_background = torch.tensor(B_array, dtype=torch.float32)

    anchor_idx = np.random.randint(n_samples)
    anchor_ecef = ecef_tensor[anchor_idx].unsqueeze(0)

    r_dipole = anchor_ecef + torch.tensor(np.random.uniform(-2, 2, (1, 3)), dtype=torch.float32)
    m_dipole = torch.tensor(np.random.uniform(-5, 5, (1, 3)), dtype=torch.float32)

    B_disturbance = errorSource(ecef_tensor, r_dipole, m_dipole)
    B_total = B_background + B_disturbance

    return lla_tensor, ecef_tensor, B_background, B_disturbance, B_total, r_dipole, m_dipole