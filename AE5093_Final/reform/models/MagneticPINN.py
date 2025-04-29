import torch
import torch.nn as nn

class MagneticPINN(nn.Module):
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
    
    def pdeLoss(model, X_norm, ecef_mean, ecef_std):
        X_norm.requires_grad_(True)

        B_pred = model(X_norm)

        divB_x = torch.autograd.grad(
            B_pred[:, 0].sum(), X_norm, create_graph=True, retain_graph=True
        )[0][:, 0]

        divB_y = torch.autograd.grad(
            B_pred[:, 1].sum(), X_norm, create_graph=True, retain_graph=True
        )[0][:, 1]

        divB_z = torch.autograd.grad(
            B_pred[:, 2].sum(), X_norm, create_graph=True, retain_graph=True
        )[0][:, 2]

        divB = divB_x + divB_y + divB_z
        return torch.mean(divB**2)