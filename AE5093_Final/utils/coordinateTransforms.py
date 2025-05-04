import numpy as np

def lla2ecef(lat, lon, alt):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)

    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + alt) * np.sin(lat)

    return np.stack((X, Y, Z), axis=-1)

def ecef2lla(X, Y, Z):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)

    lon = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)
    lat = np.arctan2(Z, p * (1 - e2))
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)

    return lat, lon, alt