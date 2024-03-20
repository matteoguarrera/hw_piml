import torch

from darcyflow import darcy_residuals_no_forcing

def data_fitting_loss(predicted: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute the data fitting loss.
    :param predicted: torch.Tensor - predicted solution (B x ...)
    :param target: torch.Tensor - target solution (B x ...)
    :param reduction: str - reduction method for the loss. None returns a tensor of batch size
    :return: torch.Tensor - data fitting loss
    """
    # print('Target shape:', target.shape)

    difference = predicted - target
    batch_size = difference.shape[0]
    difference = difference.view(batch_size, -1)
    loss = torch.linalg.norm(difference, dim=1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def rel_data_loss(predicted: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
    """
    Compute the relative data loss.
    :param predicted: torch.Tensor - predicted solution (B x ...)
    :param target: torch.Tensor - target solution (B x ...)
    :param reduction: str - reduction method for the loss. None returns a tensor of batch size
    :return: torch.Tensor - relative data loss
    """
    difference = predicted - target
    batch_size = difference.shape[0]
    difference = difference.view(batch_size, -1)
    loss = torch.linalg.norm(difference, dim=1) / torch.linalg.norm(target.view(batch_size, -1), dim=1)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def darcy_pde_residual(u: torch.Tensor, mesh: torch.Tensor, diffusion: torch.Tensor, reduction: str = 'mean'):
    """
    Compute the PDE residual loss of solving the Darcy equation.
    :param u: torch.Tensor - solution of the Darcy equation (B x Nx x Nt x n)
    :param mesh: torch.Tensor - mesh (B x Nx x Nt x 2)
    :param diffusion: torch.Tensor - diffusion coefficient (B x Nx x Nt x 1)
    :param reduction: str - reduction method for the loss. None returns a tensor of batch size
    :return: torch.Tensor - PDE residual loss
    """
    residuals_no_forcing = darcy_residuals_no_forcing(u, mesh, diffusion)
    return data_fitting_loss(residuals_no_forcing, torch.ones_like(residuals_no_forcing), reduction=reduction)
