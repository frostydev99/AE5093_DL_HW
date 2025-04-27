import torch
import torch.nn as nn
import torch.autograd as autograd

from utils.earthDipole import earthDipoleField

class MagneticPINN(nn.Module):
    def __init__(self):
        super(MagneticPINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)
    
def totalLoss(model, coords):

    coords.requires_grad = True

    B_pred = model(coords)

    Bx, By, Bz = B_pred[:, 0], B_pred[:, 1], B_pred[:, 2]

    grads = torch.ones_like(Bx, device=coords.device)

    dBx_dx = autograd.grad(Bx, coords, grad_outputs=grads, create_graph=True)[0][:, 0]
    dBy_dy = autograd.grad(By, coords, grad_outputs=grads, create_graph=True)[0][:, 1]
    dBz_dz = autograd.grad(Bz, coords, grad_outputs=grads, create_graph=True)[0][:, 2]

    div_B = dBx_dx + dBy_dy + dBz_dz
    loss_div_B = torch.mean(div_B ** 2)

    ideal_B = earthDipoleField(coords)
    loss_dipole = torch.mean((B_pred - ideal_B) ** 2)

    return loss_div_B + 10 * loss_dipole