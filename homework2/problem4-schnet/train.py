import torch
import numpy as np
import argparse
from schnet import SchNet
import torch_geometric.nn.models as geometric_models

from lmdb_dataset import LmdbDataset, data_list_collater
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import L2MAELoss, rotate_3d_coordinates
import wandb


def train(data_dir, size, r_max, batch_size, lr, max_epochs, device, original_model):

    # # W&B Run
    # wandb.login()
    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="dl-physics-hw2-schnet",
    #     name=f"aspirin_{size}_r_max={r_max}",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": lr,
    #         "epochs": max_epochs,
    #         "batch_size": batch_size,
    #         "r_max": r_max,
    #         "size": size
    #     }
    # )

    # data
    train_dir = f"{data_dir}/{size}/train"
    val_dir = f"{data_dir}/{size}/val"
    test_dir = f"{data_dir}/{size}/test"
    train_dataset = LmdbDataset({'src': train_dir})
    val_dataset = LmdbDataset({'src': val_dir})
    test_dataset = LmdbDataset({'src': test_dir})
    train_dataloader = DataLoader(train_dataset, collate_fn=data_list_collater, \
                                  batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, collate_fn=data_list_collater, \
                                batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_list_collater, \
                                 batch_size=batch_size, shuffle=False)
    # model
    if original_model:
        model = geometric_models.SchNet(hidden_channels=128, num_filters=128,
                                       num_interactions=6, num_gaussians=50,
                                       cutoff=r_max, max_num_neighbors=32,
                                       readout='add', dipole=False,
                                       mean=None, std=None, )
    else:
        model = SchNet(cutoff=r_max)
    model = model.to(device)
    torch.save(model.state_dict(), 'schnet_model_pretrain.pt')

    # optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
    force_loss = L2MAELoss()

    """
        Batch size is 128, it seems that the first molecules have 21 atoms each.
        [batch.y]     Ground truth energy
        [batch.pos]   Ground truth position
        [batch.atomic_numbers]    Concatenated atomic numbers, of all the batch
        [batch.force]    Ground truth of forces per atom
        [batch.batch]    Categorical indices of molecules, we are considering a batch of 128
        
        [batch.natoms]    Number of atoms per molecule
        [batch.cell]       Seems to be ignored, in the forward pass arguments.
    """

    epoch = 0
    while epoch < max_epochs and optimizer.param_groups[0]['lr'] > 1e-7:
        model.train()
        print(f"Train Epoch {epoch + 1}")
        for batch in tqdm(train_dataloader):
            # TODO: fill in the training loop
            # print(batch)
            """
            # Create a custom batch to check that the forces are equivariant                                           
            degree_rot = np.random.rand(3, 1) * 360

            rotated_pos = rotate_3d_coordinates(original_pos,
                                                x_degrees=degree_rot[0],
                                                y_degrees=degree_rot[1],
                                                z_degrees=degree_rot[2])

            rotated_forces = rotate_3d_coordinates(batch.force,
                                                   x_degrees=degree_rot[0],
                                                   y_degrees=degree_rot[1],
                                                   z_degrees=degree_rot[2])
            """
            # batch.pos.requires_grad = True

            optimizer.zero_grad()
            out_energy, out_forces = model.forward(batch.atomic_numbers.to(device),
                                                   batch.pos.to(device), batch=batch.batch.to(device))

            # To understand what's inside
            # for name, value in batch:
            #     if value.dtype == torch.long:
            #         print(name, torch.max(value))

            loss = force_loss(out_forces, batch.force.to(device))
            loss.backward()
            optimizer.step()

            # print({"train_force_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
            wandb.log({"train_force_loss": loss, "lr": optimizer.param_groups[0]['lr']})

        model.eval()
        print(f"Val Epoch {epoch + 1}")
        for batch in val_dataloader:
            # TODO: fill in validation loop

            # out_energy, out_forces = model.forward(batch.atomic_numbers.to(device),
            #                                        batch.pos.to(device),
            #                                        batch=batch.batch.to(device))
            # batch.pos.requires_grad = True

            out_energy, out_forces = model.forward(batch.atomic_numbers.to(device),
                                batch.pos.to(device), batch=batch.batch.to(device))

            loss = force_loss(out_forces,
                              batch.force.to(device))
            # loss = energy_loss(pred_energy, batch['y'])

            mean_val_loss = loss.mean().item()
            wandb.log({"val_force_loss": loss})
        scheduler.step(mean_val_loss)

        epoch += 1

    # Final Test
    test_losses = []
    for batch in test_dataloader:
        # TODO: fill in test loop
        # out_energy, out_forces = model.forward(batch.atomic_numbers.to(device),
        #                                        batch.pos.to(device),
        #                                        batch=batch.batch.to(device))
        batch.pos.requires_grad = True

        out_energy, out_forces = model.forward(batch.atomic_numbers.to(device),
                            batch.pos.to(device), batch=batch.batch.to(device))

        loss = force_loss(out_forces,
                          batch.force.to(device))

        test_losses.append(loss.mean().item())
    wandb.log({"test_force_loss": sum(test_losses) / len(test_losses)})
    torch.save(model.state_dict(), 'schnet_model.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--size', type=str, default='1k', help='Size of dataset')
    parser.add_argument('--r_max', type=float, default=5.0, help='Cutoff (A) for radius graph construction')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default="aspirin", help='Directory of data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs')

    # parser.add_argument('--geo_model', type=bool, default=False, help='Use geometric model')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # print('DEBUG MODE')
    train(args.data_dir, args.size, args.r_max, args.batch_size, args.lr, args.max_epochs, device, args.geo_model)
