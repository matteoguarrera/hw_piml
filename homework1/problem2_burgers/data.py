import torch
from torch.utils.data import Dataset
import numpy as np
from utils import MatReader


class BurgersDataset(Dataset):
    def __init__(self, data_path, train=True):
        super().__init__()
        self.train = train
        dataloader = MatReader(data_path, to_cuda=torch.cuda.is_available())
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.ic_vals = dataloader.read_field("input")  # Initial conditions
        self.solutions = dataloader.read_field("output")  # Solution
        self.tspan = dataloader.read_field("tspan")

        nx = self.ic_vals.shape[1]
        nt = self.tspan.shape[1]

        # Discretized Grid (Nx, Nt, 2)
        self.grid = torch.from_numpy(
            np.mgrid[0: 1: 1 / nx, 0: 1: 1 / nt]).permute(1, 2, 0).to(self.device)

        # Quirk of data: Need to transpose x, t
        self.solutions = self.solutions.permute(0, 2, 1)

    def __len__(self):
        # TODO
        pass

    def __getitem__(self, idx):
        # TODO
        pass
