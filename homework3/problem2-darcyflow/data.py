import torch
import typing
from torch.utils.data import Dataset
import numpy as np
import h5py as h5
import scipy


def darcy_twodgrid(num_x, num_y, bot=(0, 0), top=(1, 1)):
    """
    Create (x, y) spatial grid for Darcy Flow.

    Args:
        num_x: points in the x direction
        num_y: points in the y direction
        bot: min for x/y
        top: max for x/y
    """
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = np.linspace(x_bot, x_top, num_x)
    y_arr = np.linspace(y_bot, y_top, num_y)
    xx, yy = np.meshgrid(x_arr, y_arr)
    mesh = np.stack([xx, yy], axis=2)
    return mesh

class DarcyFlowDataset(Dataset):
    """
    Dataset for Darcy Flow data.
    """
    def __init__(self, path: str, device: torch.device, args, debug=False):
        n_samples_x = 241
        subsample_freq = 6
        self.subsampled_n_x = int(n_samples_x // subsample_freq) + 1
        data = scipy.io.loadmat(path)
        coeffs = data["coeff"]
        solutions = data["sol"]
        n_samples = coeffs.shape[0]
        # We subsample to learn on a smaller grid
        self.coeffs = torch.from_numpy(coeffs[:n_samples, ::subsample_freq, ::subsample_freq]).to(device).to(args.precision)
        self.solutions = torch.from_numpy(solutions[:n_samples, ::subsample_freq, ::subsample_freq]).to(device).to(args.precision)
        # Mesh is (Nx, Ny, 2) where the 2 points correspond to x/y
        # All meshes are the same
        self.mesh = torch.from_numpy(darcy_twodgrid(
            self.subsampled_n_x, self.subsampled_n_x
        )).to(device).to(args.precision)

        if debug:
            self.solutions = self.solutions[:debug]
            self.coeffs = self.coeffs[:debug]
    def __len__(self) -> int:
        return self.coeffs.shape[0]
    
    def __getitem__(self, idx: int):
        """
        Mesh[..., 0] is the x coordinate and Mesh[..., 1] is the y coordinate.
        Diffusion coefficients and solutions are scalar valued functions defined on the mesh.
        :return: torch.Tensor, torch.Tensor, torch.Tensor - mesh (Nx, Ny, 2), diffusion coefficients (Nx, Ny, 1), solution (Nx, Ny, 1)
        """
        return self.mesh, self.coeffs[idx, ..., None],  self.solutions[idx][..., None]
