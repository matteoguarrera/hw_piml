import torch
import torch.nn.functional as nnF
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data import BurgersDataset
from model import ConvNet2D
from pde import burgers_pde_residual, burgers_data_loss
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    burgers_train = BurgersDataset(
        'data/Burgers_train_1000_visc_0.01.mat', train=True)
    burgers_validation = BurgersDataset(
        'data/Burgers_test_50_visc_0.01.mat', train=False)

    # Hyperparameters
    lr = 5e-3
    batch_size = 16
    epochs = 40

    # Setup optimizer, model, data loader etc.
    # TODO

    # Training Loop
    # TODO
    for epoch in tqdm(range(epochs)):
        pass

    # Validation Loop
    # TODO


if __name__ == '__main__':
    torch.manual_seed(0)
    train()
