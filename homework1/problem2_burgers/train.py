import torch
import torch.nn.functional as nnF
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from data import BurgersDataset
from model import ConvNet2D
from pde import burgers_pde_residual, burgers_data_loss, burgers_pde_residual_fast
from utils import evaluate_pde_residual_train

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

    # Compute spacing
    sample, _ = burgers_train[0]
    dx = sample[0, 1, :] - sample[0, 0, :]
    dt = sample[1, :, 1] - sample[1, :, 0]
    assert torch.max(dt) == torch.min(dt)
    assert torch.max(dx) == torch.min(dx)
    dx, dt = dx[0], dt[0]
    print(dx, dt)
    assert dx > 0 and dt > 0

    criterion = torch.nn.MSELoss()

    loss_list = evaluate_pde_residual_train(train_loader, fast=True)
    print(f'Mean: {np.mean(loss_list)} \t Std: {np.std(loss_list)}')

    alpha = 1.0
    loss_name = 'data_loss'  #'data_loss' # 'residual_loss' , f'combined_loss_alpha'
    print(f'Loss name: {loss_name}')
    retrain = False
    fname = f'report/model_{loss_name}_{epochs}.pth'
    fname_loss = f'report/loss_{loss_name}_{epochs}.pkl'

    if os.path.isfile(fname) and retrain == False:
        model.load_state_dict(torch.load(fname))
        model.eval()
        train_losses = pickle.load(open(fname_loss, 'rb'))
        print(f'Loaded pretrained model: {fname}')
    else:
        print(f'Training model with {loss_name}')

        optimizer = optim.Adam(model.parameters(), lr=lr)

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
                    loss = burgers_data_loss(outputs, labels)
                    residual = burgers_pde_residual_fast(dx, dt, outputs)
                    loss += criterion(residual, torch.zeros_like(residual))

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_losses.append(loss.item())
            print(f'Epoch: {epoch} {running_loss} {loss.item()}, {np.mean(train_losses[-5:])}')

        # ONLY DATA LOSS  Epoch: 39  1300.1681728363037, 20.431795234680177
        # Evaluate dataset on pde loss: Mean: 0.10332986782849865 	 Std: 0.19170095194512526
        # is dx 0.0099? is dt - 0.0099?
        torch.save(model.state_dict(), fname)
        pickle.dump(np.array(train_losses), open(fname_loss, 'wb'), pickle.HIGHEST_PROTOCOL)

    print(f'Avg training loss: {np.mean(train_losses):.3f}')
    print(f'Std training loss: {np.std(train_losses):.3f}')

    print('Done training')
    # Visualize the training loss
    fig, (ax1) = plt.subplots(1,1, figsize=(10, 4))
    ax1.plot(train_losses, label='Training Loss')
    ax1.grid()
    plt.suptitle(f'{loss_name}', fontsize= 14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    plt.savefig(f'report/training_{loss_name}.pdf')
    plt.show()

    # Evaluate the model on the test dataset and visualize predictions for the first 3 instances
    model.eval()

    sample_test, target_test = next(iter(test_loader))
    outputs_test = model(sample_test)

    fig, ax = plt.subplots(2, 3, figsize=(10, 8))
    for i in range(3):

        ax[0, i].imshow(outputs_test[i].detach().cpu().numpy())
        ax[0, i].set_title(f'[sample # {i}] Prediction')

        ax[1, i].imshow(target_test[i].detach().cpu().numpy())
        ax[1, i].set_title(f'[sample # {i}] Ground Truth')


    plt.tight_layout()
    plt.suptitle(f'{loss_name}', fontsize= 14)
    plt.savefig(f'report/test_set_images_{loss_name}.pdf')
    plt.show()

    # Validation Loop
    # TODO
    with torch.no_grad():
        test_losses = []
        for i, (inputs, labels) in enumerate(test_loader):

            outputs = model(inputs)

            data_loss = burgers_data_loss(outputs, labels)
            residual = burgers_pde_residual_fast(dx, dt, outputs)
            residual_loss = criterion(residual, torch.zeros_like(residual))

            test_losses.append((data_loss.item(), residual_loss.item()))

        test_losses = np.array(test_losses)

    print(f'Avg Test Data loss: {np.mean(test_losses[:, 0]):.5f}')
    print(f'Std Test Data loss: {np.std(test_losses[:, 0]):.5f}')

    print(f'Avg Test Residual loss: {np.mean(test_losses[:, 1]):.5f}')
    print(f'Std Test Residual loss: {np.std(test_losses[:, 1]):.8f}')

    plt.plot(test_losses[:, 0], label='[Test] Data Loss')
    plt.plot(test_losses[:, 1], label='[Test] Residual Loss')
    plt.grid()
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{loss_name}')
    plt.savefig(f'report/validation_losses_model_trained_{loss_name}.pdf')
    plt.show()

    print('Done')
    return

if __name__ == '__main__':
    torch.manual_seed(0)
    train()
