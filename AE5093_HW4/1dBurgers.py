import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# === Seed & Device ===
seed = 45
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Hyperparameters ===
numEpochs = 5000
learningRate = 0.001
numInt = 5000
numIC = 256
numBC = 256

# === Model ===
class PINN1DBurgers(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# === Dataset Generators ===
def generate_uniform_pts(num):
    x = torch.rand((num, 1)) * 2 - 1  # x -> [-1, 1]
    t = torch.rand((num, 1))          # t -> [0, 1]
    return x.to(device), t.to(device)

def generate_initial_conditions(num):
    x = torch.linspace(-1, 1, num).view(-1, 1)
    t = torch.zeros_like(x)
    u = -torch.sin(np.pi * x)
    return x.to(device), t.to(device), u.to(device)

def generate_boundary_conditions(num):
    t = torch.linspace(0, 1, num).view(-1, 1)
    x_left = -torch.ones_like(t)
    x_right = torch.ones_like(t)
    u = torch.zeros_like(t)
    
    return (
        torch.cat([x_left, x_right]).to(device),
        torch.cat([t, t]).to(device),
        torch.cat([u, u]).to(device)
    )

# === Loss Functions ===
def pde_loss(model, x, t, nu):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    res = u_t + u * u_x - nu * u_xx
    return torch.mean(res**2)

# Initial Conditions
def initial_loss(model, x_ic, t_ic, u_ic):
    u_pred = model(x_ic, t_ic)
    return torch.mean((u_pred - u_ic) ** 2)

# Boundary Conditions
def boundary_loss(model, x_bc, t_bc, u_bc):
    u_pred = model(x_bc, t_bc)
    return torch.mean((u_pred - u_bc) ** 2)

# Learning Rate Annealing
def computeLambdaHat(loss):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1).detach().abs())

    grads = torch.cat(grads)
    return grads.max() / grads.norm()

# Allocate memory for loss history
dataLossHist  = []
pdeLossHist   = []
totalLossHist = []

# Allocate memory for lambda history
lambda_ic_hist = []
lambda_bc_hist = []
lambda_pde_hist = []

# === Training ===
def train(model, optimizer, nu, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, useAnnealing=False):
    # === Initialize dynamic weights ===
    lambda_ic = torch.tensor(1.0).to(device)
    lambda_bc = torch.tensor(1.0).to(device)
    lambda_pde = torch.tensor(1.0).to(device)
    alpha = 0.9

    # === Initialize loss history ===
    dataLossHist.clear()
    pdeLossHist.clear()
    totalLossHist.clear()
    lambda_ic_hist.clear()
    lambda_bc_hist.clear()
    lambda_pde_hist.clear()

    for epoch in range(numEpochs):
        optimizer.zero_grad()

        loss_pde = pde_loss(model, x_int, t_int, nu)
        loss_ic = initial_loss(model, x_ic, t_ic, u_ic)
        loss_bc = boundary_loss(model, x_bc, t_bc, u_bc)

        if epoch > 1 and useAnnealing:
            lambda_ic_hat = computeLambdaHat(loss_ic)
            lambda_bc_hat = computeLambdaHat(loss_bc)
            lambda_pde_hat = computeLambdaHat(loss_pde)

            # Update lambda using moving average
            lambda_ic = alpha * lambda_ic + (1 - alpha) * lambda_ic_hat
            lambda_bc = alpha * lambda_bc + (1 - alpha) * lambda_bc_hat
            lambda_pde = alpha * lambda_pde + (1 - alpha) * lambda_pde_hat

        loss_data = lambda_ic * loss_ic + lambda_bc * loss_bc
        total_loss = lambda_pde * loss_pde +  loss_data

        # Store loss history
        dataLossHist.append(loss_data.item())
        pdeLossHist.append(loss_pde.item())
        totalLossHist.append(total_loss.item())

        # Store lambda history
        lambda_ic_hist.append(lambda_ic.item())
        lambda_bc_hist.append(lambda_bc.item())
        lambda_pde_hist.append(lambda_pde.item())

        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"[nu={nu:.5f}] Epoch {epoch:>4}: "
                  f"Total={total_loss.item():.6f}, PDE={loss_pde.item():.6f}, "
                  f"Data={loss_data.item():.6f}")
            
            print(f"lambda_ic: {lambda_ic.item():.6f}, "
                  f"lambda_bc: {lambda_bc.item():.6f}, "
                  f"lambda_pde: {lambda_pde.item():.6f}")

# === Inference & Plot ===
def plot_solution(model, title="Predicted u(x,t)"):
    x = torch.linspace(-1, 1, 256).view(-1, 1).to(device)
    t = torch.linspace(0, 1, 256).view(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(256, 256)
 
    # === Plot Prediction ===
    plt.figure(figsize=(8, 6))
    plt.contourf(X.cpu(), T.cpu(), u_pred, levels=100, cmap='plasma')
    plt.colorbar(label='u(x,t)')
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # === Plot Prediction ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.cpu().numpy(), T.cpu().numpy(), u_pred, cmap='plasma', edgecolor='none')
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.set_zlabel("u_pred")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

    # === Plot Loss History ===
    plt.figure(figsize=(8, 6))
    plt.semilogy(dataLossHist, label='Data Loss')
    plt.semilogy(pdeLossHist, label='PDE Loss')
    plt.semilogy(totalLossHist, label='Total Loss')
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# === Main Loop ===
if __name__ == "__main__":
    nu_values = [0.01 / np.pi, 0.0001 / np.pi, 0.0]

    for nu in nu_values:
        print(f"\n--- Training for nu = {nu:.6f} ---")

        model = PINN1DBurgers().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

        x_int, t_int = generate_uniform_pts(numInt)
        x_ic, t_ic, u_ic = generate_initial_conditions(numIC)
        x_bc, t_bc, u_bc = generate_boundary_conditions(numBC)

        train(model, optimizer, nu, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, useAnnealing=False)

        model_path = f"pinn_burgers_nu{nu:.6f}.pth"
        torch.save(model.state_dict(), model_path)

        plot_solution(model, title=f"Predicted u(x,t), nu = {nu:.6f}")