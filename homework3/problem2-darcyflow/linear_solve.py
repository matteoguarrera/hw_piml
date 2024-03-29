import torch
from torch import autograd
from functools import partial
from torch.func import vjp

def solve_linear(A, b):
    """
    Solve the normal linear system A^T A x = A^T b. A need not be square.
    :param A: torch.Tensor - matrix A (B x N x M)
    :param b: torch.Tensor - vector b (B x M)
    :return: torch.Tensor - solution x (B x N)
    """
    # assert A.shape[2] == b.shape[1] # documentation is wrong

    # We solve the normal form to avoid the need to compute the pseudo-inverse
    A_T = A.permute(0, 2, 1)
    normal_b = torch.bmm(A_T, b[..., None]).squeeze()
    normal_A = torch.bmm(A_T, A)
    # Regularize to prevent singular matrices
    normal_A += 5e-3 * torch.eye(normal_A.shape[1], device=normal_A.device, dtype=normal_A.dtype)  
    w = LinearSolve.apply(normal_A, normal_b)
    return w

class LinearSolve(autograd.Function):
    
    @staticmethod
    def forward(ctx, A, b):
        """
        Solve the linear system A x = b. A is a batch of matrices and b is a batch of vectors.
        A is guaranteed to be square. 
        NOTE: Use torch.linalg.solve to solve the linear system.
        :param ctx: autograd context
        :param A: torch.Tensor - matrix A (B x N x N)    # PSI
        :param b: torch.Tensor - vector b (B x N)        # ONE,   1
        :return: torch.Tensor - solution x (B x N)       # omega
        """
        # TODO 
        omega = torch.linalg.solve(A, b)
        ctx.save_for_backward(A, b, omega)

        assert omega.shape == b.shape

        return omega
        
    @staticmethod
    def backward(ctx, upstream_grad):
        """
        Compute the gradient of the linear solve with respect to A and b.
        :param ctx: autograd context
        :param upstream_grad: torch.Tensor - upstream gradient
        :return: torch.Tensor (B x N x N), torch.Tensor (B x N) - gradient of the linear solve with respect to A and b
        """
        # TODO

        PSI, ONE, omega = ctx.saved_tensors
        # def optimality_cond(PSI_, ONE_, omega_):
        #     return PSI_ @ omega_.T - ONE_
        #
        # javrev_fn = torch.func.jacrev(optimality_cond, argnums=2)
        # jacrev_batched = torch.func.vmap(javrev_fn)
        # PSI_der = jacrev_batched(PSI, ONE, omega)
        #
        # def optimality_fn(PSI_, ONE_):
        #     return torch.einsum('b m n, b n -> bm', PSI_, omega) - ONE_
        #
        # v = - upstream_grad
        # # u = torch.linalg.solve(PSI_der, v.unsqueeze(-1)).squeeze(-1)
        # u = torch.linalg.solve(torch.transpose(PSI_der, 1, 2), v.unsqueeze(-1)).squeeze(-1)
        #
        # evaluations, vpj_fn = vjp(optimality_fn, PSI, ONE)
        #
        # return vpj_fn(u)


        #################### Mine ####################
        #A = PSI

        # def optimality_cond(_PSI, _ONE):
        #     return torch.bmm(_PSI, omega.unsqueeze(-1)).squeeze() - _ONE
        #
        # u = torch.linalg.solve(A, -upstream_grad)
        #
        # evaluations, funct_ = vjp(optimality_cond, PSI, ONE )
        #
        # return funct_(u)

        #A = PSI

        def optimality_cond(_PSI, _ONE):
            return torch.bmm(_PSI, omega.unsqueeze(-1)).squeeze() - _ONE

        u = torch.linalg.solve(-torch.transpose(PSI, 1, 2), upstream_grad)

        evaluations, funct_ = vjp(optimality_cond, PSI, ONE)

        return funct_(u)


