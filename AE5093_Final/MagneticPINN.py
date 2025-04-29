import torch
import torch.nn as nn

class MagneticPINN_BField(nn.Module):
    """
    Predicts B-field [Bx, By, Bz] at a 3D input position [x, y, z]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)  # Output: Bx, By, Bz
        )

    def forward(self, x):
        return self.net(x)

def pdeLoss_BField(model, X_norm, ecef_mean, ecef_std):
    """
    Enforces ∇·B = 0 using autograd. Input X is normalized ECEF positions.
    """
    X_norm.requires_grad_(True)

    B_pred = model(X_norm)  # depends on X_norm

    divB = 0
    for i in range(3):  # Bx, By, Bz
        grad_i = torch.autograd.grad(
            B_pred[:, i].sum(), X_norm,  # must match input to model
            create_graph=True, retain_graph=True
        )[0][:, i]
        divB += grad_i

    return torch.mean(divB**2)
