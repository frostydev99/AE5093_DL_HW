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

    m_dot_r = torch.sum(m_dipole * r, dim=1, keepdim=True)
    B = (mu_0 / (4 * np.pi)) * (3 * r_hat * m_dot_r - m_dipole) / (r_norm**3)

    return B

# === Data Generation ===
def generateTrainingData(n_samples):
    # === Generate random geographic positions ===
    lat = np.random.uniform(-60, 60, n_samples)
    lon = np.random.uniform(-180, 180, n_samples)
    alt = np.random.uniform(0, 1000, n_samples)
    lla = np.stack([lat, lon, alt], axis=1)
    lla_tensor = torch.tensor(lla, dtype=torch.float32)

    # === Convert to ECEF ===
    ecef = lla2ecef(lat, lon, alt)
    ecef_tensor = torch.tensor(ecef, dtype=torch.float32)

    # === Get WMM background fields ===
    B_array = np.array([getWMMField(lat[i], lon[i], alt[i]) for i in range(n_samples)])
    B_background = torch.tensor(B_array, dtype=torch.float32)

    # === For each point, generate a unique dipole nearby ===
    # Random dipole offset: each source is near the measurement point
    r_dipole_offsets = np.random.uniform(-500, 500, size=(n_samples, 3))  # in meters
    r_dipole = ecef_tensor + torch.tensor(r_dipole_offsets, dtype=torch.float32)

    # Random dipole moments
    m_dipole = torch.tensor(np.random.uniform(-5, 5, (n_samples, 3)), dtype=torch.float32)

    # === Compute Dipole Disturbances Individually ===
    B_disturbance = errorSource(ecef_tensor, r_dipole, m_dipole)

    # === Total field is background + disturbance ===
    B_total = B_background + B_disturbance

    # === Save R Sensor ===
    r_sensor = ecef_tensor

    return lla_tensor, ecef_tensor, B_background, B_disturbance, B_total, r_dipole, m_dipole, r_sensor
