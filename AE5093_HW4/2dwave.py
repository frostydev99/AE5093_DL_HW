import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# === Seed & Device ===
seed = 45
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

# === Hyperparameters ===
numEpochs = 10000
learningRate = 0.0005
numInt = 5000
numIC = 256
numBC = 256

# === PDE Parameters ===
kCases = [2, 10, 25]
c = 1.0

# === Rowdy Activation ===
class Rowdy(nn.Module):
    def __init__(self, features):
        super().__init__()
        # Parameters for phi_2 and phi_3
        self.w = nn.Parameter(torch.randn(features))
        self.b = nn.Parameter(torch.randn(features))
        self.v = nn.Parameter(torch.randn(features))
        self.c = nn.Parameter(torch.randn(features))
        self.alpha = nn.Parameter(torch.ones(features) * 0.1)
        self.beta = nn.Parameter(torch.ones(features) * 0.1)

    def forward(self, x):
        # Base activation
        phi1 = torch.tanh(x)
        # Perturbations
        phi2 = self.alpha * torch.cos(self.w * x + self.b)
        phi3 = self.beta * torch.sin(self.v * x + self.c)
        return phi1 + phi2 + phi3
    
class PINN2DWave_Rowdy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            Rowdy(50),
            nn.Linear(50, 50),
            Rowdy(50),
            nn.Linear(50, 1)
        )

    def forward(self, x, y, t):
        return self.net(torch.cat([x,y,t], dim=1))

# === Model ===
class PINN2DWave(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y, t):
        return self.net(torch.cat([x,y,t], dim=1))
    
# === Dataset Generators ===
def generate_points(n_int, n_bdy, k):
    # Interior: x, y ∈ [-1, 1], t ∈ [0, 1]
    x_int = torch.rand((n_int, 1), device=device, requires_grad=True) * 2 - 1
    y_int = torch.rand((n_int, 1), device=device, requires_grad=True) * 2 - 1
    t_int = torch.rand((n_int, 1), device=device, requires_grad=True)

    # Boundary (square edges)
    t_bdy = torch.rand((n_bdy // 4, 1))  # Random times on all edges

    xb = torch.cat([
        torch.rand((n_bdy // 4, 1)) * 2 - 1,  # x on bottom edge
        torch.rand((n_bdy // 4, 1)) * 2 - 1,  # x on top edge
        -torch.ones((n_bdy // 4, 1)),         # x = -1 (left edge)
        torch.ones((n_bdy // 4, 1))           # x = 1 (right edge)
    ])

    yb = torch.cat([
        -torch.ones((n_bdy // 4, 1)),         # y = -1 (bottom)
        torch.ones((n_bdy // 4, 1)),          # y = 1 (top)
        torch.rand((n_bdy // 4, 1)) * 2 - 1,  # y left edge
        torch.rand((n_bdy // 4, 1)) * 2 - 1   # y right edge
    ])

    tb = torch.cat([t_bdy, t_bdy, t_bdy, t_bdy])

    omega = np.sqrt(2) * c * np.pi * k

    ub = torch.sin(k*np.pi*xb) * torch.sin(k*np.pi*yb) * torch.cos(omega*tb)

    return x_int.to(device), y_int.to(device), t_int.to(device), xb.to(device), yb.to(device), tb.to(device), ub.to(device)

def generate_initial_conditions(num_points_per_dim, k):
    x = torch.linspace(-1, 1, num_points_per_dim)
    y = torch.linspace(-1, 1, num_points_per_dim)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    T_flat = torch.zeros_like(X_flat)

    # Initial condition: u(x, y, 0) = sin(kπx) sin(kπy)
    U_flat = torch.sin(k * np.pi * X_flat) * torch.sin(k * np.pi * Y_flat)

    return X_flat.to(device), Y_flat.to(device), T_flat.to(device), U_flat.to(device)

# === Loss Functions ===
# == PDE Loss ==
def pdeLoss(model, x, y, t, k):
    u = model(x, y, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True)[0]

    u_pred = u_tt - np.pow(c, 2) * (u_xx + u_yy)

    omega = np.sqrt(2)*c*np.pi*k

    u_exact = torch.sin(k*np.pi*x) * torch.sin(k*np.pi*y) * torch.cos(omega*t)

    return torch.mean((u_pred)**2)

# == Boundary Loss ==
def boundary_loss(model, x, y, t, u):
    ub_pred = model(x, y, t)

    return torch.mean((ub_pred - u)**2)

# == Initial Condition Loss ==
def intitial_loss(model, x, y, t, u):
    ui_pred = model(x, y, t)

    return torch.mean((ui_pred - u) ** 2)

# Allocate memory for loss history
dataLossHist  = []
pdeLossHist   = []
totalLossHist = []

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

def train(model, optimizer, k, x_int, y_int, t_int,
          x_ic, y_ic, t_ic, u_ic,
          x_bc, y_bc, t_bc, u_bc,
          useAnnealing=False):

    dataLossHist.clear()
    pdeLossHist.clear()
    totalLossHist.clear()

    lambda_ic = torch.tensor(1.0).to(device)
    lambda_bc = torch.tensor(1.0).to(device)
    lambda_pde = torch.tensor(1.0).to(device)
    alpha = 0.9

    model.train()

    for epoch in range(numEpochs):
        optimizer.zero_grad()

        # === PDE Loss ===
        loss_pde = pdeLoss(model, x_int, y_int, t_int, k)

        # === Initial Condition Loss ===
        loss_ic = intitial_loss(model, x_ic, y_ic, t_ic, u_ic)

        # === Boundary Condition Loss ===
        loss_bc = boundary_loss(model, x_bc, y_bc, t_bc, u_bc)

        # === Annealing ===
        if epoch > 1 and useAnnealing:
            lambda_ic_hat = computeLambdaHat(loss_ic)
            lambda_bc_hat = computeLambdaHat(loss_bc)
            lambda_pde_hat = computeLambdaHat(loss_pde)

            # Update lambda using moving average
            lambda_ic = alpha * lambda_ic + (1 - alpha) * lambda_ic_hat
            lambda_bc = alpha * lambda_bc + (1 - alpha) * lambda_bc_hat
            lambda_pde = alpha * lambda_pde + (1 - alpha) * lambda_pde_hat

        loss_data = lambda_ic * loss_ic + lambda_bc * loss_bc
        total_loss = lambda_pde * loss_pde + loss_data

        # Record loss history
        dataLossHist.append((loss_ic + loss_bc).item())
        pdeLossHist.append(loss_pde.item())
        totalLossHist.append(total_loss.item())

        # Backprop and step
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, PDE Loss: {loss_pde.item():.4f}, "
                  f"IC Loss: {loss_ic.item():.4f}, BC Loss: {loss_bc.item():.4f}") 

# === Inference & Plot ===
def plot_truth_pred_error(model, k, t_val=0.25):
    model.eval()

    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = torch.linspace(-1, 1, 100).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)
    T_flat = torch.full_like(X_flat, t_val).to(device)

    omega = np.sqrt(2) * c * np.pi * k
    u_true = torch.sin(k * np.pi * X_flat) * torch.sin(k * np.pi * Y_flat) * torch.cos(omega * T_flat)
    with torch.no_grad():
        u_pred = model(X_flat, Y_flat, T_flat)

    u_true = u_true.cpu().numpy().reshape(100, 100)
    u_pred = u_pred.cpu().numpy().reshape(100, 100)
    error = np.abs(u_true - u_pred)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # True
    c0 = axs[0].contourf(X.cpu().numpy(), Y.cpu().numpy(), u_true, levels=50, cmap='viridis')
    axs[0].set_title(f'True Solution at t={t_val}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(c0, ax=axs[0])

    # Predicted
    c1 = axs[1].contourf(X.cpu().numpy(), Y.cpu().numpy(), u_pred, levels=50, cmap='viridis')
    axs[1].set_title(f'Predicted Solution at t={t_val}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(c1, ax=axs[1])

    # Error
    c2 = axs[2].contourf(X.cpu().numpy(), Y.cpu().numpy(), error, levels=50, cmap='inferno')
    axs[2].set_title(f'Absolute Error at t={t_val}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    fig.colorbar(c2, ax=axs[2])

    plt.tight_layout()
    plt.show()

def plot_solution(model, k):

    # === Plot Loss History ===
    plt.figure(figsize=(8, 6))
    plt.semilogy(dataLossHist, label='Data Loss')
    plt.semilogy(pdeLossHist, label='PDE Loss')
    plt.semilogy(totalLossHist, label='Total Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss History - k = {k}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plot_contour_slices(model, k)

    plot_truth_pred_error(model, k)

    animate_solution(model, k)

def animate_solution(model, k, num_frames=100):
    model.eval()

    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = torch.linspace(-1, 1, 100).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)

    # Estimate z limits over all frames (optional but more robust)
    z_max = -float('inf')
    z_min = float('inf')
    for frame in range(num_frames):
        t_val = torch.full_like(X_flat, frame / num_frames).to(device)
        with torch.no_grad():
            u_pred = model(X_flat, Y_flat, t_val).cpu().numpy().reshape(100, 100)
        z_max = max(z_max, np.max(u_pred))
        z_min = min(z_min, np.min(u_pred))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        t_val = torch.full_like(X_flat, frame / num_frames).to(device)
        with torch.no_grad():
            u_pred = model(X_flat, Y_flat, t_val).cpu().numpy().reshape(100, 100)

        ax.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), u_pred, cmap=cm.viridis)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f't = {frame / num_frames:.2f} - k = {k}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x, y, t)')

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    plt.show()

def plot_contour_slices(model, k, num_slices=6):
    model.eval()

    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = torch.linspace(-1, 1, 100).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)

    times = torch.linspace(0, 1, num_slices)
    
    fig, axs = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))

    for i, t_val in enumerate(times):
        T_flat = torch.full_like(X_flat, t_val.item()).to(device)

        with torch.no_grad():
            u_pred = model(X_flat, Y_flat, T_flat).cpu().numpy().reshape(100, 100)

        # --- Contour plot ---
        ax1 = axs[0, i]
        contour = ax1.contourf(X.cpu().numpy(), Y.cpu().numpy(), u_pred, levels=50, cmap='viridis')
        ax1.set_title(f't = {t_val:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        fig.colorbar(contour, ax=ax1)

        # --- Diagonal slice plot ---
        ax2 = axs[1, i]
        diag_x = torch.linspace(-1, 1, 100).reshape(-1, 1).to(device)
        diag_y = diag_x
        diag_t = torch.full_like(diag_x, t_val.item()).to(device)

        with torch.no_grad():
            diag_u = model(diag_x, diag_y, diag_t).cpu().numpy()

        ax2.plot(diag_x.cpu().numpy(), diag_u)
        ax2.set_title(f'Diagonal Slice x=y, t={t_val:.2f}')
        ax2.set_xlabel('x=y')
        ax2.set_ylabel('u(x,x,t)')

    plt.tight_layout()
    plt.suptitle(f'Stacked Contours and Diagonal Slices - k = {k}', fontsize=16, y=1.02)
    plt.show()

# === Main Loop ===
if __name__ == "__main__":
    # === Initialize Model and Optimizer ===
    model = PINN2DWave().to(device)
    # model = PINN2DWave_Rowdy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # === Train for each k case ===
    for k in kCases:
        print(f"Training for k = {k}")
        
        # === Generate Initial and Boundary Conditions ===
        x_int, y_int, t_int, x_bc, y_bc, t_bc, u_bc = generate_points(numInt, numBC, k)
        x_ic, y_ic, t_ic, u_ic = generate_initial_conditions(numIC, k)

        train(model, optimizer, k,
              x_int, y_int, t_int,
              x_ic, y_ic, t_ic, u_ic,
              x_bc, y_bc, t_bc, u_bc, useAnnealing=True)

        plot_solution(model, k)