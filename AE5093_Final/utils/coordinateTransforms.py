import numpy as np
import torch

def lla_to_ecef(lat, lon, alt):
    """
    Converts (lat, lon, alt) in degrees, meters to ECEF coordinates.
    """
    a = 6378137.0  # Earth semi-major axis
    e = 8.1819190842622e-2  # eccentricity

    lat = np.radians(lat)
    lon = np.radians(lon)

    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e**2) + alt) * np.sin(lat)

    return np.stack((x, y, z), axis=-1)

def normalize_positions(positions):
    """ Normalize ECEF positions for NN training """
    mean = np.mean(positions, axis=0)
    std = np.std(positions, axis=0)
    norm_positions = (positions - mean) / std
    return norm_positions, mean, std

def unnormalize_positions(norm_positions, mean, std):
    return norm_positions * std + mean
