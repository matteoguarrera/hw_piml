import torch
import typing


def __central_differences_grad__(dx: torch.Tensor, dy: torch.Tensor, u: torch.Tensor):
    # Compute finite differences for interior points
    # Calculate gradient of u with respect to the mesh

    uu_xm1, uu_xp0, uu_xp1 = u[:, :-2, 1:-1], u[:, 1:-1, 1:-1], u[:, 2:, 1:-1]
    uu_ym1, uu_yp0, uu_yp1 = u[:, 1:-1, :-2], u[:, 1:-1, 1:-1], u[:, 1:-1, 2:]

    u_x = (uu_xp1 - uu_xm1) / (2 * dx)
    # u_xx = (uu_xp1 - 2 * uu_xp0 + uu_xm1) / (dx ** 2)
    # u_y = (uu_ym1 - uu_yp1) / (2 * dy)
    u_y = (uu_yp1 - uu_ym1) / (2 * dy)

    return u_x, u_y




def darcy_residuals_no_forcing(u: torch.Tensor, mesh: torch.Tensor, diffusion_coeff: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the residuals of the Darcy equation without forcing term.
    n is the number of basis functions. When n=1, we compute the residuals as normal.
    :param u: torch.Tensor - solution of the Darcy equation (B x Nx x Ny x n)
    :param mesh: torch.Tensor - mesh (B x Nx x Ny x 2)
    :param diffusion_coeff: torch.Tensor - diffusion coefficient (B x Nx x Ny x n)
    """
    # TODO
    n = diffusion_coeff.shape[3]
    assert n == 1
    # u, solution of the darcy equation.  u = b.T @ omega
    # Extracting the dx, dy as in hw 1 for computing the central finite differences

    # Compute spacing
    # assuming equispaced points, otherwise the all exercise wouldn't work.
    # The spacing is also the first point.
    dx = mesh[0, 1, 0, 1]
    dy = mesh[0, 0, 1, 0]

    # Calculate gradient of u with respect to the mesh, gradient is a vector
    u_x, u_y = __central_differences_grad__(dx, dy, u)
    # These are resized by 2. From 41x41 to 39x39

    # Multiply the gradient by the diffusion coefficient
    diff_coeff_resized = diffusion_coeff[:, 1:-1, 1:-1, :]  # need to be resized
    vu_x = diff_coeff_resized * u_x
    vu_y = diff_coeff_resized * u_y

    # Computing the gradients again, for the divergence
    # Mesh is still the same so dx, dy dont change.
    # https://en.wikipedia.org/wiki/Divergence

    u_xx, _ = __central_differences_grad__(dx, dy, vu_x)
    _, u_yy = __central_differences_grad__(dx, dy, vu_y)

    # Sum the components of the flux to get the divergence
    residuals = - (u_xx + u_yy)

    # The residuals are the divergence of the flux
    assert residuals.shape[1:-1] == (37, 37)

    return residuals  #



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
