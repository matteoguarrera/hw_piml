import torch
import torch.nn.functional as nnF


def burgers_pde_residual(x, t, u):
    # x: (B, Nx, Nt)
    # t: (B, Nx, Nt)
    # u: (B, Nx, Nt)

    nu = 0.01

    # TODO
    pass


def burgers_data_loss(predicted, target):
    # Relative L2 Loss
    # Predicted: (B, Nx, Nt)
    # Target: (B, Nx, Nt)
    # TODO
    pass
