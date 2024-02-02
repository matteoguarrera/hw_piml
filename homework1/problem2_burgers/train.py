import torch
import torch.nn.functional as nnF
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data import BurgersDataset
from model import ConvNet2D
from pde import burgers_pde_residual, burgers_data_loss, burgers_pde_residual_fast
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
    epochs = 2 #40

    # Setup optimizer, model, data loader etc.
    # TODO
    # Create DataLoader objects for training and testing datasets
    train_loader = torch.utils.data.DataLoader(burgers_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(burgers_validation, batch_size=batch_size)
    model = ConvNet2D().to(device)
    sample, target = next(iter(train_loader))

    outputs = model(sample)  # check if it works
    print(outputs.size())

    # Compute spacing
    dx = sample[0, 0, 1, :] - sample[0, 0, 0, :]
    dt = sample[0, 1, :, 1] - sample[0, 1, :, 0]
    assert torch.max(dt) == torch.min(dt)
    assert torch.max(dx) == torch.min(dx)
    dx, dt = dx[0], dt[0]
    assert dx > 0 and dt > 0

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    alpha = 1.0
    loss_name = 'residual_loss'  #'data_loss' # 'residual_loss' , f'combined_loss_alpha'

    # Training Loop
    # TODO
    train_losses = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            if loss_name == 'data_loss':
                loss = burgers_data_loss(outputs, labels)
            elif loss_name == 'residual_loss':
                residual = burgers_pde_residual_fast(dx, dt, outputs)
                loss = criterion(residual, torch.zeros_like(residual))
            elif loss_name == 'combined_loss_alpha':
                pass

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_losses.append(loss.item())
        print(f'Epoch: {epoch} {running_loss} {loss.item()}, {np.mean(train_losses[-5:])}')

    # ONLY DATA LOSS  Epoch: 39  1300.1681728363037, 20.431795234680177
    # Evaluate dataset on pde loss: Mean: 0.10332986782849865 	 Std: 0.19170095194512526
    # is dx 0.0099? is dt - 0.0099?


    # Visualize the training loss
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(5, 15))
    ax1.plot(train_losses, label='Training Loss')
    ax1.grid()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_losses[-10:], label='Training Loss')
    ax2.grid()
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.savefig(f'training_{loss_name}.pdf')
    plt.show()
    # Evaluate the model on the test dataset and visualize predictions for the first 3 instances
    model.eval()

    sample_test, target_test = next(iter(test_loader))
    outputs_test = model(sample_test)

    fig, ax = plt.subplots(3, 2, figsize=(5, 3))
    for i in range(3):

        ax[i, 0].imshow(outputs_test[i].detach().cpu().numpy())
        ax[i, 0].set_title(f'[sample # {i}] Prediction')

        ax[i, 1].imshow(target_test[i].detach().cpu().numpy())
        ax[i, 1].set_title(f'[sample # {i}] Ground Truth')

    plt.savefig('test_set_images.pdf')
    plt.show()

    # Validation Loop
    # TODO


    with torch.no_grad():
        test_losses = []
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= 3:
                break
            outputs = model(inputs)
            print(outputs.size())


            residual = burgers_pde_residual_fast(dx, dt, outputs)
            loss = criterion(residual, torch.zeros_like(residual))

            test_losses.append(loss.item())
    plt.plot(test_losses, label='Test Loss')

if __name__ == '__main__':
    torch.manual_seed(0)
    train()
