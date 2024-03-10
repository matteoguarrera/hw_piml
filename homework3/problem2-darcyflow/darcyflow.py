import torch
import typing


def darcy_residuals_no_forcing(u: torch.Tensor, mesh: torch.Tensor, diffusion_coeff: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the residuals of the Darcy equation without forcing term.
    n is the number of basis functions. When n=1, we compute the residuals as normal.
    :param u: torch.Tensor - solution of the Darcy equation (B x Nx x Ny x n)
    :param mesh: torch.Tensor - mesh (B x Nx x Ny x 2)
    :param diffusion_coeff: torch.Tensor - diffusion coefficient (B x Nx x Ny x n)
    """
    # TODO
    pass

def mollify(u: torch.Tensor, mesh: torch.Tensor, scale=1e-3) -> torch.Tensor:
    """
    Mollify the solution of the Darcy equation.
    :param u: torch.Tensor - solution of the Darcy equation (B x Nx x Ny x n)
    :param mesh: torch.Tensor - mesh (B x Nx x Ny x 2)
    :param scale: float - scale of the mollifier
    :return: torch.Tensor - mollified solution
    """
    mollifier = (
        scale * torch.sin(torch.pi * mesh[..., 0]) * torch.sin(torch.pi * mesh[..., 1])
    )
    return u * mollifier[..., None]
