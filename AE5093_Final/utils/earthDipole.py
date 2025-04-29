import torch
import numpy as np
from wmm2020 import wmm

def earthDipoleField(coords):
    m = torch.tensor([0, 0, 7.94e22], device=coords.device)
    mu_0 = 4 * torch.pi * 1e-7  # Vacuum permeability

    r = torch.norm(coords, dim=1, keepdim=True) + 1e-8
    r_hat = coords / r

    m_dot_r = torch.sum(m * coords, dim=1, keepdim=True)

    t1 = (3 * r_hat * m_dot_r) / (r ** 2)
    t2 = m / r

    B_Tesla = (mu_0 / (4 * torch.pi)) * (t1 - t2) / r**2

    B_Gauss = B_Tesla * 1e4  # Convert Tesla to Gauss

    return B_Gauss

def earthWMMField(coords):
    """
    coords: (N, 3) tensor in ECEF meters
    Returns Earth's magnetic field at those points in Gauss
    """
    # === ECEF to LLA (needed because WMM is based on lat/lon/alt)
    def ecef_to_lla(ecef):
        a = 6378137.0
        e = 8.1819190842622e-2
        x, y, z = ecef[:,0], ecef[:, 1], ecef[:, 2]
        b = np.sqrt(a**2 * (1 - e**2))
        ep = np.sqrt((a**2 - b**2) / b**2)
        p = np.sqrt(x**2 + y**2)
        th = np.arctan2(a * z, b * p)
        lon = np.arctan2(y, x)
        lat = np.arctan2((z + ep**2 * b * np.sin(th)**3),
                         (p - e**2 * a * np.cos(th)**3))
        N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.degrees(lat)
        lon = np.degrees(lon)
        return lat, lon, alt

    ecef_np = coords.detach().cpu().numpy()
    lat, lon, alt = ecef_to_lla(ecef_np)

    B_list = []
    for la, lo, al in zip(lat, lon, alt):
        B_ned = wmm(glats=la, glons=lo, alt_km=al / 1000.0, yeardec=2025.0)

        # Correct field components:
        B_north = B_ned['north'] * 1e-5  # [nT] -> [Gauss]
        B_east  = B_ned['east'] * 1e-5
        B_down  = B_ned['down'] * 1e-5

        

        # Append in NED order (North, East, Down)
        B_list.append([B_north, B_east, B_down])

    B_gauss = torch.tensor(B_list, dtype=torch.float32, device=coords.device)

    return B_gauss
