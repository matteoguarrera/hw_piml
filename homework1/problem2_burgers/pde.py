import torch
import torch.nn.functional as nnF


def burgers_pde_residual(dx, dt, u):
    # x: (B, Nx, Nt)
    # t: (B, Nx, Nt)
    # u: (B, Nx, Nt)
    '''Actually I don't need x, t. I just pass dx, dt'''

    nu = 0.01
    # TODO
    ''' [START] Automatic differentiation 
        A possibility might be using automatic differentiation to compute the derivatives
        Assuming u is a function of x and t, represented as a neural network
    '''
    # u.requires_grad_(True)
    #
    # # Compute derivatives
    # u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    # u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    # u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    #
    # # Compute the residual of Burgers' equation
    # residual = u_t + u * u_x - nu * u_xx

    '''[END] Automatic differentiation '''

    # x: (B, Nx, Nt)
    # t: (B, Nx, Nt)
    # u: (B, Nx, Nt)

    nu = 0.01
    # TODO
    ''' [START] Automatic differentiation 
        A possibility might be using automatic differentiation to compute the derivatives
        Assuming u is a function of x and t, represented as a neural network
    '''
    # u.requires_grad_(True)
    #
    # # Compute derivatives
    # u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    # u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    # u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    #
    # # Compute the residual of Burgers' equation
    # residual = u_t + u * u_x - nu * u_xx

    '''[END] Automatic differentiation '''

    # Calculate spacings
    # The space-temporal spacing is uniform. It's around .0099

    # print(dx, dt)
    # Initialize the residual array
    residual = torch.zeros_like(u)

    '''
    Implement the code as vectorial. Speed up the computation.
    # x_tp1 = x[:, :, 2:] # t forward
    # x_tm1 = x[:, :,  :-2]  # t backward
    # x_t = x[:, :, 1:-1]
    '''

    # Compute finite differences for interior points
    for b in range(u.shape[0]):  # Loop over batches
        for i in range(1, u.shape[1] - 1):  # Loop over spatial dimension
            for j in range(1, u.shape[2] - 1):  # Loop over temporal dimension
                u_x = (u[b, i + 1, j] - u[b, i - 1, j]) / (2 * dx)
                u_xx = (u[b, i + 1, j] - 2 * u[b, i, j] + u[b, i - 1, j]) / dx ** 2
                u_t = (u[b, i, j + 1] - u[b, i, j - 1]) / (2 * dt)

                # print(u_t.shape, u.shape, u_x.shape)
                # Compute the residual for Burgers' equation
                residual[b, i, j] = u_t + u[b, i, j] * u_x - nu * u_xx

    # What about boundary conditions?
    # Should we handle it?

    return residual


def burgers_pde_residual_fast(dx, dt, u):
    def __per_sample_fast__(uu):
        # [1:-1] is used to remove the first and the last
        # It is done on the matrix xp0 and tp0
        uu_xm1, uu_xp0, uu_xp1 = uu[:-2, 1:-1], uu[1:-1, 1:-1], uu[2:, 1:-1]
        uu_tm1, uu_tp0, uu_tp1 = uu[1:-1, :-2], uu[1:-1, 1:-1], uu[1:-1, 2:]

        u_x = (uu_xp1 - uu_xm1) / (2 * dx)
        u_xx = (uu_xp1 - 2 * uu_xp0 + uu_xm1) / (dx ** 2)
        u_t = (uu_tp1 - uu_tm1) / (2 * dt)

        residual_sample = u_t + uu[1:-1, 1:-1] * u_x - nu * u_xx
        return residual_sample

    # Initialize the residual array
    #  = torch.zeros_like(u)
    nu = 0.01

    # Compute finite differences for interior points
    # res = [__per_sample__(u[b]) for b in range(u.shape[0])]   # Loop over batches
    res = [__per_sample_fast__(u[b]).unsqueeze(0) for b in range(u.shape[0])]  # Loop over batches
    residual = torch.concatenate(res, dim=0)

    return residual

def burgers_data_loss(predicted, target):
    # Relative L2 Loss
    # Predicted: (B, Nx, Nt)
    # Target: (B, Nx, Nt)
    # TODO
    output = torch.linalg.norm(predicted - target, dim = (1,2))/torch.linalg.norm(target, dim=(1, 2))  #
    output = torch.sum(output)
    return output

