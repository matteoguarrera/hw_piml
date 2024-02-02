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
            np.mgrid[0: 1: 1 / nx, 0: 1: 1 / nt]).permute(1, 2, 0).to(self.device).float()

        # Quirk of data: Need to transpose x, t
        self.solutions = self.solutions.permute(0, 2, 1)

        # Custom code
        # To speed up unsqueeze and broadcast here the initial conditions
        self.ic_broadcast = torch.broadcast_to(self.ic_vals.unsqueeze(-1),
                                               (*self.ic_vals.shape, 101)).unsqueeze(-1).float()

    def __len__(self):
        # TODO
        return self.solutions.shape[0]

    def __getitem__(self, idx):
        # TODO
        # Broadcast initial condition
        # I guess we should permute it since conv2D input is (N,Cin,H,W)
        return torch.cat((self.grid, self.ic_broadcast[idx]), axis=-1).permute(2,0,1), self.solutions[idx]  # input cat initial condition