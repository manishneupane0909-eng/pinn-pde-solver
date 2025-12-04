"""
Physics-Informed Neural Network (PINN) for solving Burgers' equation.

Burgers' equation: ∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²)

This implementation uses PyTorch to train a neural network that satisfies
the PDE constraints directly in the loss function.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class PINN(nn.Module):
    """Physics-Informed Neural Network for Burgers' equation."""
    
    def __init__(self, layers=[2, 50, 50, 50, 1], activation=nn.Tanh):
        """
        Initialize PINN.
        
        Args:
            layers: List of layer sizes [input_dim, hidden1, ..., output_dim]
            activation: Activation function class
        """
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation()
    
    def forward(self, x, t):
        """Forward pass through the network."""
        # Concatenate x and t as input
        X = torch.cat([x, t], dim=1)
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            X = self.activation(layer(X))
        
        # Output layer (no activation)
        u = self.layers[-1](X)
        return u


class BurgersPINN:
    """PINN solver for Burgers' equation."""
    
    def __init__(self, nu=0.01/np.pi, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize solver.
        
        Args:
            nu: Viscosity parameter (default from literature)
            device: 'cuda' or 'cpu'
        """
        self.nu = nu
        self.device = torch.device(device)
        self.model = None
        self.loss_history = []
        
    def build_model(self, layers=[2, 50, 50, 50, 1]):
        """Build and initialize the neural network."""
        self.model = PINN(layers).to(self.device)
        return self.model
    
    def u_t(self, x, t):
        """Compute ∂u/∂t using automatic differentiation."""
        u = self.model(x, t)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        return u_t
    
    def u_x(self, x, t):
        """Compute ∂u/∂x using automatic differentiation."""
        u = self.model(x, t)
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        return u_x
    
    def u_xx(self, x, t):
        """Compute ∂²u/∂x² using automatic differentiation."""
        u_x = self.u_x(x, t)
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        return u_xx
    
    def pde_residual(self, x, t):
        """
        Compute PDE residual: ∂u/∂t + u(∂u/∂x) - ν(∂²u/∂x²)
        
        This should be zero if the PDE is satisfied.
        """
        u = self.model(x, t)
        u_t = self.u_t(x, t)
        u_x = self.u_x(x, t)
        u_xx = self.u_xx(x, t)
        
        residual = u_t + u * u_x - self.nu * u_xx
        return residual
    
    def ic_loss(self, x_ic, t_ic, u_ic):
        """Initial condition loss: u(x, 0) = -sin(πx)"""
        u_pred = self.model(x_ic, t_ic)
        return torch.mean((u_pred - u_ic)**2)
    
    def bc_loss(self, x_bc, t_bc):
        """Boundary condition loss: u(-1, t) = u(1, t) = 0"""
        u_pred = self.model(x_bc, t_bc)
        return torch.mean(u_pred**2)
    
    def pde_loss(self, x_pde, t_pde):
        """PDE residual loss (physics-informed term)"""
        residual = self.pde_residual(x_pde, t_pde)
        return torch.mean(residual**2)
    
    def total_loss(self, x_ic, t_ic, u_ic, x_bc, t_bc, x_pde, t_pde, 
                   lambda_ic=10.0, lambda_bc=10.0):
        """
        Total loss combining IC, BC, and PDE terms.
        
        Args:
            lambda_ic: Weight for initial condition loss
            lambda_bc: Weight for boundary condition loss
        """
        loss_ic = self.ic_loss(x_ic, t_ic, u_ic)
        loss_bc = self.bc_loss(x_bc, t_bc)
        loss_pde = self.pde_loss(x_pde, t_pde)
        
        total = lambda_ic * loss_ic + lambda_bc * loss_bc + loss_pde
        return total, loss_ic, loss_bc, loss_pde
    
    def generate_training_data(self, n_ic=100, n_bc=50, n_pde=10000):
        """
        Generate training data points.
        
        Args:
            n_ic: Number of initial condition points
            n_bc: Number of boundary condition points
            n_pde: Number of collocation points for PDE
        """
        # Initial condition: u(x, 0) = -sin(πx) for x ∈ [-1, 1]
        x_ic = np.random.uniform(-1, 1, (n_ic, 1))
        t_ic = np.zeros((n_ic, 1))
        u_ic = -np.sin(np.pi * x_ic)
        
        # Boundary conditions: u(-1, t) = u(1, t) = 0 for t ∈ [0, 1]
        t_bc = np.random.uniform(0, 1, (n_bc, 1))
        x_bc_left = -np.ones((n_bc//2, 1))
        x_bc_right = np.ones((n_bc//2, 1))
        x_bc = np.vstack([x_bc_left, x_bc_right])
        t_bc = np.vstack([t_bc[:n_bc//2], t_bc[n_bc//2:]])
        
        # Collocation points for PDE: (x, t) ∈ [-1, 1] × [0, 1]
        x_pde = np.random.uniform(-1, 1, (n_pde, 1))
        t_pde = np.random.uniform(0, 1, (n_pde, 1))
        
        # Convert to tensors
        x_ic = torch.FloatTensor(x_ic).to(self.device).requires_grad_(True)
        t_ic = torch.FloatTensor(t_ic).to(self.device).requires_grad_(True)
        u_ic = torch.FloatTensor(u_ic).to(self.device)
        
        x_bc = torch.FloatTensor(x_bc).to(self.device).requires_grad_(True)
        t_bc = torch.FloatTensor(t_bc).to(self.device).requires_grad_(True)
        
        x_pde = torch.FloatTensor(x_pde).to(self.device).requires_grad_(True)
        t_pde = torch.FloatTensor(t_pde).to(self.device).requires_grad_(True)
        
        return x_ic, t_ic, u_ic, x_bc, t_bc, x_pde, t_pde
    
    def train(self, epochs=10000, lr=0.001, verbose=True):
        """
        Train the PINN.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print training progress
        """
        if self.model is None:
            self.build_model()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Generate training data
        x_ic, t_ic, u_ic, x_bc, t_bc, x_pde, t_pde = self.generate_training_data()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute loss
            loss, loss_ic, loss_bc, loss_pde = self.total_loss(
                x_ic, t_ic, u_ic, x_bc, t_bc, x_pde, t_pde
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Store loss history
            self.loss_history.append({
                'total': loss.item(),
                'ic': loss_ic.item(),
                'bc': loss_bc.item(),
                'pde': loss_pde.item()
            })
            
            # Print progress
            if verbose and (epoch % 1000 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
                      f"IC: {loss_ic.item():.6f} | BC: {loss_bc.item():.6f} | "
                      f"PDE: {loss_pde.item():.6f}")
    
    def predict(self, x, t):
        """Predict u(x, t) using the trained model."""
        x_tensor = torch.FloatTensor(x).to(self.device)
        t_tensor = torch.FloatTensor(t).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            u = self.model(x_tensor, t_tensor)
        self.model.train()
        
        return u.cpu().numpy()
    
    def plot_solution(self, save_path='results.png'):
        """Visualize the solution."""
        # Create spatial and temporal grids
        x = np.linspace(-1, 1, 100)
        t = np.linspace(0, 1, 50)
        X, T = np.meshgrid(x, t)
        
        # Predict solution
        u_pred = self.predict(X.flatten()[:, None], T.flatten()[:, None])
        U = u_pred.reshape(X.shape)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 3D surface plot
        ax1 = axes[0]
        surf = ax1.contourf(X, T, U, levels=50, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('PINN Solution: u(x, t)')
        plt.colorbar(surf, ax=ax1)
        
        # Time snapshots
        ax2 = axes[1]
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t_idx = np.argmin(np.abs(t - t_val))
            ax2.plot(x, U[t_idx, :], label=f't = {t_val:.2f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x, t)')
        ax2.set_title('Solution at Different Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Solution plot saved to {save_path}")
        plt.close()
    
    def plot_loss_history(self, save_path='loss_history.png'):
        """Plot training loss history."""
        if not self.loss_history:
            print("No loss history available. Train the model first.")
            return
        
        epochs = range(len(self.loss_history))
        total_loss = [h['total'] for h in self.loss_history]
        ic_loss = [h['ic'] for h in self.loss_history]
        bc_loss = [h['bc'] for h in self.loss_history]
        pde_loss = [h['pde'] for h in self.loss_history]
        
        plt.figure(figsize=(12, 6))
        plt.semilogy(epochs, total_loss, label='Total Loss', linewidth=2)
        plt.semilogy(epochs, ic_loss, label='IC Loss', alpha=0.7)
        plt.semilogy(epochs, bc_loss, label='BC Loss', alpha=0.7)
        plt.semilogy(epochs, pde_loss, label='PDE Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss history plot saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train PINN for Burgers\' equation')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--nu', type=float, default=0.01/np.pi, help='Viscosity parameter')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine device
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize solver
    solver = BurgersPINN(nu=args.nu, device=device)
    solver.build_model()
    
    print(f"\nTraining PINN for {args.epochs} epochs...")
    print(f"Viscosity parameter ν = {args.nu:.6f}\n")
    
    # Train
    solver.train(epochs=args.epochs, lr=args.lr)
    
    # Visualize results
    print("\nGenerating visualizations...")
    solver.plot_solution(save_path=output_dir / 'solution.png')
    solver.plot_loss_history(save_path=output_dir / 'loss_history.png')
    
    # Save model
    model_path = output_dir / 'model.pth'
    torch.save(solver.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

