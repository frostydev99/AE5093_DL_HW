import torch
import numpy as np
from wmm2020 import wmm

def get_WMM_field(lat, lon, alt):
    """
    Gets Earth's magnetic field at given lat, lon, alt (all floats).
    Returns magnetic field vector (North, East, Down) in Tesla
    """
    # WMM expects lat, lon in degrees, altitude in kilometers
    field = wmm(lat, lon, alt/1000.0, 2025)
    B_N = field['north'].data
    B_E = field['east'].data
    B_D = field['down'].data

    return np.array([B_N, B_E, B_D]) * 1e-9  # Convert from nT to Tesla

def generateTrainingData(n_samples):
    """
    Generates synthetic training data for a magnetic field model.
    :param n_samples: Number of samples to generate
    :return: Tuple of (LLA coordinates, magnetic field)
    """

    # Generate random LLA coordinates
    latitudes = np.random.uniform(-90, 90, n_samples)   # degrees
    longitudes = np.random.uniform(-180, 180, n_samples)  # degrees
    altitudes = np.random.uniform(0, 10000, n_samples)   # altitude in meters (0m to 10km)

    lla_coords = np.stack([latitudes, longitudes, altitudes], axis=1)
    lla_tensor = torch.tensor(lla_coords, dtype=torch.float32)

    # Compute magnetic field at each location
    B_background_list = []
    for i in range(n_samples):
        B = get_WMM_field(latitudes[i], longitudes[i], altitudes[i])
        B_background_list.append(B)

    B_background_tensor = torch.tensor(np.array(B_background_list), dtype=torch.float32)

    # Generate random magnetic field error in Teslas (small disturbance)
    B_error = np.random.uniform(-1e-8, 1e-8, (n_samples, 3))
    B_error_tensor = torch.tensor(B_error, dtype=torch.float32)

    # Final magnetic field = Background + small random error
    B_measured_tensor = B_background_tensor + B_error_tensor

    return lla_tensor, B_measured_tensor
