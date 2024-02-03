import torch
import scipy.io
import h5py
import numpy as np
from tqdm import tqdm

from pde import burgers_pde_residual_fast, burgers_pde_residual

def evaluate_pde_residual_train(train_loader, fast=True):
    sample, target = next(iter(train_loader))

    # Compute spacing
    dx = sample[0, 0, 1, :] - sample[0, 0, 0, :]
    dt = sample[0, 1, :, 1] - sample[0, 1, :, 0]
    assert torch.max(dt) == torch.min(dt)
    assert torch.max(dx) == torch.min(dx)
    dx, dt = dx[0], dt[0]
    assert dx > 0 and dt > 0
    print('dx, dt: ', dx, dt )
    criterion = torch.nn.MSELoss()

    loss_list = []
    for sample, target in tqdm(train_loader):
        if fast:
            residual = burgers_pde_residual_fast(dx, dt, target)
        else:
            residual = burgers_pde_residual(dx, dt, target)

        loss = criterion(residual, torch.zeros_like(residual))
        loss_list.append(loss.item())
    return loss_list


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float
