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
learningRate = 0.0002
numInt = 5000
numIC = 512
numBC = 512

# === PDE Parameters ===
kCases = [25]
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
        # Sinusoidal Rowdyness
        phi2 = self.alpha * torch.cos(self.w * x + self.b)
        phi3 = self.beta * torch.sin(self.v * x + self.c)
        return phi1 + phi2 + phi3
    
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
    
# === Model with Rowdy ===
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
        return self.net(torch.cat([x, y, t], dim=1))
    
# === Dataset Generators ===
def generate_points(n_int, n_bdy, k):
    # Interior Points
    x_int = torch.rand((n_int, 1), device=device, requires_grad=True) * 2 - 1 # x -> [-1, 1]
    y_int = torch.rand((n_int, 1), device=device, requires_grad=True) * 2 - 1 # y -> [-1, 1]
    t_int = torch.rand((n_int, 1), device=device, requires_grad=True)         # t -> [0, 1]

    # Boundary (square edges)
    t_bdy = torch.rand((n_bdy // 4, 1))

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
    T_flat = torch.zeros_like(X_flat, requires_grad=True)

    # Initial condition: u(x, y, 0) = sin(k pi x) sin(k pi y) cos(omega t) (t = 0)
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

    return torch.mean((u_pred)**2)

# == Boundary Loss ==
def boundary_loss(model, x_b, y_b, t_b, u_b):
    ub_pred = model(x_b, y_b, t_b)

    return torch.mean((ub_pred - u_b)**2)

# == Initial Condition Loss ==
def intitial_loss(model, x_ic, y_ic, t_ic, u_ic):
    ui_pred = model(x_ic, y_ic, t_ic)

    return torch.mean((ui_pred - u_ic) ** 2)

# Allocate memory for loss history
dataLossHist  = []
pdeLossHist   = []
totalLossHist = []

# Allocate memory for lambda history
lambdaHistIC = []
lambdaHistBC = []
lambdaHistPDE = []

# Learning Rate Annealing
def computeLambdaHat(loss):
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

    lambdaHistIC.clear()
    lambdaHistBC.clear()
    lambdaHistPDE.clear()

    lambda_ic = torch.tensor(1.0).to(device)
    lambda_bc = torch.tensor(0.5).to(device)
    lambda_pde = torch.tensor(0.5).to(device)
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

        # Record lambda history
        lambdaHistIC.append(lambda_ic.item())
        lambdaHistBC.append(lambda_bc.item())
        lambdaHistPDE.append(lambda_pde.item())

        # Backprop and step
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, PDE Loss: {loss_pde.item():.4f}, "
                  f"IC Loss: {loss_ic.item():.4f}, BC Loss: {loss_bc.item():.4f}") 
            
            print(f"Lambda IC: {lambda_ic.item():.4f}, Lambda BC: {lambda_bc.item():.4f}, Lambda PDE: {lambda_pde.item():.4f}")
            print("=========================================")

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

    c0 = axs[0].contourf(X.cpu().numpy(), Y.cpu().numpy(), u_true, levels=50, cmap='viridis')
    axs[0].set_title(f'True Solution at t={t_val}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(c0, ax=axs[0])

    c1 = axs[1].contourf(X.cpu().numpy(), Y.cpu().numpy(), u_pred, levels=50, cmap='viridis')
    axs[1].set_title(f'Predicted Solution at t={t_val}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(c1, ax=axs[1])

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

    # === Plot Lambda History ===
    plt.figure(figsize=(8, 6))
    plt.plot(lambdaHistIC, label='Lambda IC')
    plt.plot(lambdaHistBC, label='Lambda BC')
    plt.plot(lambdaHistPDE, label='Lambda PDE')
    plt.xlabel("Epoch")
    plt.ylabel("Lambda")
    plt.title(f"Lambda History - k = {k}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plot_truth_pred_error_multiple_times(model, k, times=[0.0, 0.5, 1.0])

    animate_solution(model, k)

def animate_solution(model, k, num_frames=100):
    model.eval()

    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = torch.linspace(-1, 1, 100).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)

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

def plot_truth_pred_error_multiple_times(model, k, times=[0.0, 0.25, 0.5, 0.75, 1.0]):
    model.eval()

    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = torch.linspace(-1, 1, 100).reshape(-1, 1)
    X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1).to(device)
    Y_flat = Y.reshape(-1, 1).to(device)

    omega = np.sqrt(2) * c * np.pi * k

    n_times = len(times)
    fig, axs = plt.subplots(3, n_times, figsize=(4 * n_times, 12))

    for idx, t_val in enumerate(times):
        T_flat = torch.full_like(X_flat, t_val).to(device)

        # Exact solution
        u_true = torch.sin(k * np.pi * X_flat) * torch.sin(k * np.pi * Y_flat) * torch.cos(omega * T_flat)
        u_true = u_true.cpu().numpy().reshape(100, 100)

        # Prediction
        with torch.no_grad():
            u_pred = model(X_flat, Y_flat, T_flat).cpu().numpy().reshape(100, 100)

        # Error
        error = np.abs(u_true - u_pred)

        # --- Plot True ---
        c0 = axs[0, idx].contourf(X.cpu().numpy(), Y.cpu().numpy(), u_true, levels=50, cmap='viridis')
        axs[0, idx].set_title(f'True t={t_val:.2f}')
        axs[0, idx].set_xlabel('x')
        axs[0, idx].set_ylabel('y')
        fig.colorbar(c0, ax=axs[0, idx])

        # --- Plot Prediction ---
        c1 = axs[1, idx].contourf(X.cpu().numpy(), Y.cpu().numpy(), u_pred, levels=50, cmap='viridis')
        axs[1, idx].set_title(f'Predicted t={t_val:.2f}')
        axs[1, idx].set_xlabel('x')
        axs[1, idx].set_ylabel('y')
        fig.colorbar(c1, ax=axs[1, idx])

        # --- Plot Error ---
        c2 = axs[2, idx].contourf(X.cpu().numpy(), Y.cpu().numpy(), error, levels=50, cmap='inferno')
        axs[2, idx].set_title(f'Error t={t_val:.2f}')
        axs[2, idx].set_xlabel('x')
        axs[2, idx].set_ylabel('y')
        fig.colorbar(c2, ax=axs[2, idx])

    plt.tight_layout()
    plt.suptitle(f"Truth vs Prediction vs Error (k={k})", fontsize=20, y=1.02)
    plt.show()

# === Main Loop ===
if __name__ == "__main__":

    # === Train for each k case ===
    for k in kCases:
        print(f"Training for k = {k}")

        # === Initialize Model and Optimizer ===
        # model = PINN2DWave().to(device)
        model = PINN2DWave_Rowdy().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
        
        # === Generate Initial and Boundary Conditions ===
        x_int, y_int, t_int, x_bc, y_bc, t_bc, u_bc = generate_points(numInt, numBC, k)
        x_ic, y_ic, t_ic, u_ic = generate_initial_conditions(numIC, k)

        train(model, optimizer, k,
              x_int, y_int, t_int,
              x_ic, y_ic, t_ic, u_ic,
              x_bc, y_bc, t_bc, u_bc, useAnnealing=True)

        plot_solution(model, k)