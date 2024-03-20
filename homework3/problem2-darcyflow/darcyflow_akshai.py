import torch
import typing


def first_order_finite_difference(u: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
    """Compute the first-order finite difference of a function.
    Args:
        u (torch.Tensor): function (B x Nx x Ny x n)
        mesh (torch.Tensor): mesh (B x Nx x Ny x 2)
    Returns:
        (torch.Tensor): first-order finite difference
    """
    # The last dim of mesh is the x and y coordinates
    mesh_x = mesh[:, :, :, 0]
    mesh_y = mesh[:, :, :, 1]
    # Add in a dimension to match the shape of u
    mesh_x = mesh_x.unsqueeze(3)
    mesh_y = mesh_y.unsqueeze(3)

    u_x = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (
        mesh_x[:, :, 2:, :] - mesh_x[:, :, :-2, :]
    )
    u_y = (u[:, 2:, :, :] - u[:, :-2, :, :]) / (
        mesh_y[:, 2:, :, :] - mesh_y[:, :-2, :, :]
    )

    # Trim the top and bottom of the grids to keep the shape consistent
    u_x = u_x[:, 1:-1, :, :]
    u_y = u_y[:, :, 1:-1, :]

    return u_x, u_y


def darcy_residuals_no_forcing(
    u: torch.Tensor, mesh: torch.Tensor, diffusion_coeff: torch.Tensor
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Compute the residuals of the Darcy equation without forcing term.
    n is the number of basis functions. When n=1, we compute the residuals as normal.

    Args:
        u (torch.Tensor): solution of the Darcy equation (B x Nx x Ny x n)
        mesh (torch.Tensor): mesh (B x Nx x Ny x 2)
        diffusion_coeff (torch.Tensor): diffusion coefficient (B x Nx x Ny x n)

    Returns:
        (torch.Tensor, torch.Tensor): residuals of the Darcy equation without forcing term (B x Nx x Ny x n)
    """
    # TODO:
    # Darcy Flow equation is given by:
    # -\nabla \cdot \left( \nu(x, y) \nabla u(x, y) \right) = f(x, y) = F(u)

    # Use a first-order central difference finite differences scheme to compute the partials
    u_x, u_y = first_order_finite_difference(u, mesh)

    # Multiply with the diffusion coefficient
    # Trim difficult_coeff to match the shape of u_x and u_y
    diffusion_coeff = diffusion_coeff[:, 1:-1, 1:-1, :]
    v_u_x = diffusion_coeff * u_x
    v_u_y = diffusion_coeff * u_y

    # Calcaulte the second order finite difference
    # Trim mesh to match the shape of v_u_x and v_u_y
    mesh = mesh[:, 1:-1, 1:-1, :]
    u_xx, _ = first_order_finite_difference(v_u_x, mesh)
    _, u_yy = first_order_finite_difference(v_u_y, mesh)

    return - (u_xx + u_yy)


def mollify(u: torch.Tensor, mesh: torch.Tensor, scale=1e-3) -> torch.Tensor:
    """Mollify the solution of the Darcy equation.
    Args:
        u (torch.Tensor): solution of the Darcy equation (B x Nx x Ny x n)
        mesh (torch.Tensor): mesh (B x Nx x Ny x 2)
        scale (float): scale of the mollifier

    Returns
        (torch.Tensor): mollified solution
    """
    mollifier = (
        scale * torch.sin(torch.pi * mesh[..., 0]) * torch.sin(torch.pi * mesh[..., 1])
    )
    return u * mollifier[..., None]
