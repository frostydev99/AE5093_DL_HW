import torch

def earthDipoleField(coords):
    m = torch.tensor([0,0, 7.94e22], device=coords.device)

    mu_0 = 4 * torch.pi * 1e-7

    r = torch.norm(coords, dim=1, keepdim=True) + 1e-8
    r_hat = coords / r

    m_dot_r = torch.sum(m * coords, dim=1, keepdim=True)

    t1 = (3 * r_hat * m_dot_r) / (r ** 2)

    t2 = m / r

    B = (mu_0 / (4 * torch.pi)) * (t1 - t2) / r**2

    return B