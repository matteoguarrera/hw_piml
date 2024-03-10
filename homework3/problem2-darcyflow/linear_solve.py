import torch
from torch import autograd
from functools import partial


def solve_linear(A, b):
    """
    Solve the normal linear system A^T A x = A^T b. A need not be square.
    :param A: torch.Tensor - matrix A (B x N x M)
    :param b: torch.Tensor - vector b (B x M)
    :return: torch.Tensor - solution x (B x N)
    """
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
        :param A: torch.Tensor - matrix A (B x N x N)
        :param b: torch.Tensor - vector b (B x N)
        :return: torch.Tensor - solution x (B x N)
        """
        # TODO 
        pass
        
    @staticmethod
    def backward(ctx, upstream_grad):
        """
        Compute the gradient of the linear solve with respect to A and b.
        :param ctx: autograd context
        :param upstream_grad: torch.Tensor - upstream gradient
        :return: torch.Tensor (B x N x N), torch.Tensor (B x N) - gradient of the linear solve with respect to A and b
        """
        # TODO
        pass
