# Physics-Informed Neural Network (PINN) for Burgers' Equation

PyTorch implementation of a Physics-Informed Neural Network to solve Burgers' equation. This was a project for my computational physics class.

Burgers' equation models fluid flow with nonlinear advection and diffusion:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

The PINN embeds the PDE directly into the loss function, so it learns to satisfy the physics automatically without labeled training data.

## What it does

- Trains a neural network to solve Burgers' equation
- Uses automatic differentiation to compute derivatives
- Satisfies initial and boundary conditions through the loss function
- Works on GPU for faster training

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train with default settings:
```bash
python main.py
```

Custom training:
```bash
python main.py --epochs 20000 --lr 0.0005 --nu 0.01
```

Arguments:
- `--epochs`: Number of training epochs (default: 10000)
- `--lr`: Learning rate (default: 0.001)
- `--nu`: Viscosity parameter (default: 0.01/π)
- `--device`: 'auto', 'cuda', or 'cpu' (default: 'auto')
- `--output-dir`: Where to save results (default: 'results')

## How it works

The network takes (x, t) coordinates as input and outputs u(x, t). The loss function has three parts:

1. **Initial condition loss**: Makes sure u(x, 0) = -sin(πx)
2. **Boundary condition loss**: Makes sure u(±1, t) = 0
3. **PDE residual loss**: Makes sure the PDE is satisfied everywhere

The network architecture is simple: 3 hidden layers with 50 neurons each, using tanh activation.

## Results

After training, you get:
- `solution.png`: Plot of the solution over time
- `loss_history.png`: Training loss curves
- `model.pth`: Saved model weights

The PDE residual should drop below 1e-4 after convergence. Training takes about 5-10 minutes on GPU or 20-30 minutes on CPU.

## Notes

- The viscosity parameter ν = 0.01/π creates a shock wave that diffuses over time
- Training can be sensitive to the learning rate, might need to tune it
- Higher viscosity values might need more epochs to converge
- GPU memory usage depends on how many collocation points you use

## References

Main paper I used:
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
