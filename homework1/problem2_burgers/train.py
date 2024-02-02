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
    # Create DataLoader objects for training and testing datasets
    train_loader = torch.utils.data.DataLoader(burgers_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(burgers_validation, batch_size=batch_size)
    model = ConvNet2D().to(device)
    outputs = model(next(iter(train_loader))[0])  # check if it works
    print(outputs.size())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training Loop
    # TODO
    num_epochs = 40
    train_losses = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = burgers_data_loss(outputs, labels)
            u = labels
            x = inputs[:, 0, :]  # not sure
            t = inputs[:, :, 1]  # not sure

            residual = burgers_pde_residual(x, t, u)
            loss = criterion(residual, torch.zeros_like(residual))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_losses.append(loss.item())
        print(f'Epoch: {epoch}  {running_loss}, {np.mean(train_losses[-50:])}')

    # ONLY DATA LOSS  Epoch: 39  1300.1681728363037, 20.431795234680177
    # Evaluate dataset on pde loss: Mean: 0.10332986782849865 	 Std: 0.19170095194512526
    # is dx 0.0099? is dt - 0.0099?


    # Visualize the training loss
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model on the test dataset and visualize predictions for the first 3 instances
    model.eval()

    # Validation Loop
    # TODO
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,5))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= 3:
                break
            outputs = model(inputs)
            print(outputs.size())


if __name__ == '__main__':
    torch.manual_seed(0)
    train()
