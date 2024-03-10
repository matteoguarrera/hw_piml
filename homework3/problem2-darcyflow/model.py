import torch
import torch.nn as nn
import typing
from darcyflow import darcy_residuals_no_forcing, mollify
from linear_solve import solve_linear

class ResidualLayer(nn.Module):
    """
    Simple residual layer with gelu activations. Two convolutions with one residual connection.
    """
    def __init__(self, in_places, out_places):
        super().__init__()
        self.conv1 = nn.Conv2d(in_places, out_places, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_places, out_places, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = nn.functional.gelu(out)
        out = self.conv2(out)
        out += identity
        out = nn.functional.gelu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: typing.Tuple[int], out_dim: int):
        super().__init__()
        activate_last_layer = out_dim > 1 # Check if we need to activate the last layer. Only activate in constrained case

        layers = []
        layers.append(nn.Conv2d(in_dim, hidden_dims[0], kernel_size=3, padding=1))
        layers.append(nn.GELU())
        for i in range(len(hidden_dims) - 1):
            layers.append(ResidualLayer(hidden_dims[i], hidden_dims[i+1]))
        layers.append(nn.Conv2d(hidden_dims[-1], 128, kernel_size=1, padding=0))
        layers.append(nn.Conv2d(128, out_dim, kernel_size=1, padding=0))
        if activate_last_layer:
            layers.append(nn.GELU())
        self.model = nn.Sequential(*layers)
    
    def forward(self, mesh, diffusion_coeffs) -> torch.Tensor:
        """
        :param mesh: torch.Tensor - mesh (B x Nx x Ny x 2)
        :param diffusion_coeffs: torch.Tensor - diffusion coefficients (B x Nx x Ny x 1)
        :return: torch.Tensor - output of the model (B x Nx x Ny x out_dim)
        """
        # Concatenate mesh and diffusion coefficients
        x = torch.cat([mesh, diffusion_coeffs], dim=3)
        # Normalize diffusion coefficients between 0-1
        min_val = x.reshape(x.shape[0], -1).min(dim=1).values[:, None, None, None ]
        max_val = x.reshape(x.shape[0], -1).max(dim=1).values[:, None, None, None ]
        x = (x - min_val) / (max_val - min_val)
        x = x.permute(0, 3, 1, 2)
        out = self.model(x) 
        out = out.permute(0, 2, 3, 1)
        return mollify(out, mesh)
    
class ConstrainedModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: typing.Tuple[int], n_basis_functions: int):
        super().__init__()
        # TODO 
        pass
    
    def forward(self, mesh: torch.Tensor, diffusion_coeffs: torch.Tensor) -> torch.Tensor:
        """
        :param mesh: torch.Tensor - mesh (B x Nx x Ny x 2)
        :param diffusion_coeffs: torch.Tensor - diffusion coefficients (B x Nx x Ny x 1)
        :return: torch.Tensor - output of the model (B x Nx x Ny x 1)
        """
        # TODO
        pass
    
    def setup_linear_system(self, basis_functions, mesh, diffusion_coeffs):
        """
        Setup the linear system for the constrained model.
        :param basis_functions: torch.Tensor - basis functions (B x Nx x Ny x n)
        :param mesh: torch.Tensor - mesh (B x Nx x Ny x 2)
        :param diffusion_coeffs: torch.Tensor - diffusion coefficients (B x Nx x Ny x 1)
        :return: torch.Tensor, torch.Tensor - A, b representing the linear system A\omega = b
        """
        # TODO
        pass
        