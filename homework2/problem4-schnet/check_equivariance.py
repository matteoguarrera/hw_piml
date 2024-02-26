# Verify that the model's energy and forces predictions
# are invariant and equivariant to rotation respectively.

import torch
from schnet import SchNet
from lmdb_dataset import LmdbDataset, data_list_collater
from torch.utils.data import DataLoader
from utils import L2MAELoss, rotate_3d_coordinates
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

data_dir, size = 'aspirin', '1k'
device, batch_size = 'cpu', 128
r_max = 5.0

train_dir = f"{data_dir}/{size}/train"
train_dataset = LmdbDataset({'src': train_dir})
train_dataloader = DataLoader(train_dataset, collate_fn=data_list_collater, \
                              batch_size=batch_size, shuffle=False)
# model
model = SchNet(cutoff=r_max)
model.load_state_dict(torch.load('schnet_model.pt'))

model = model.to(device)
model.eval()

force_loss = L2MAELoss()
energy_loss = torch.nn.MSELoss()

losses = {'energy': [], 'force': [], 'energy_rot': [], 'force_rot': []}

model.eval()
for batch in tqdm(train_dataloader):
    original_pos = batch.pos

    # Forward with original position
    original_pos.requires_grad = True
    out_energy, out_forces = model.forward(batch.atomic_numbers,
                                           original_pos,
                                           batch=batch.batch)

    # Rotate positions and forces
    degree_rot = np.random.rand(3, 1) * 360
    rotated_pos = rotate_3d_coordinates(original_pos, *degree_rot)
    rotated_forces = rotate_3d_coordinates(batch.force, *degree_rot)

    # Forward with rotated positions
    rotated_pos.requires_grad = True
    out_energy_rot, out_forces_rot = model.forward(batch.atomic_numbers,
                                                   rotated_pos,
                                                   batch=batch.batch)

    # [CHECK EQUIVARIANCE] Rotated forces loss should be equal to loss of non rotated.
    loss_f = force_loss(out_forces_rot, rotated_forces).detach().item()
    loss_f_rot = force_loss(out_forces, batch.force).detach().item()
    assert np.abs(loss_f - loss_f_rot) < 1e-6, print('loss_f_rot: ', loss_f_rot, '\n\n', 'loss_f: ', loss_f)

    loss_e = energy_loss(out_energy, batch.y).detach().item()
    loss_e_rot = energy_loss(out_energy_rot, batch.y).detach().item()
    assert np.abs(loss_e - loss_e_rot) < 1e-6

    losses['force'].append(loss_f)
    losses['force_rot'].append(loss_f_rot)

    losses['energy'].append(loss_e)
    losses['energy_rot'].append(loss_e_rot)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.plot(losses['force'], '*', label = 'force',)
ax1.plot(losses['force_rot'], '-', label = 'force_rotated')
ax1.legend()
ax1.grid()
ax1.set_xlabel('train batch idx')
ax1.set_ylabel('L2MAELoss')
ax1.set_title('Before training force loss')
ax2.plot(losses['energy'], '*', label = 'energy')
ax2.plot(losses['energy_rot'], '-', label = 'energy_rot')
ax2.legend()
ax2.set_xlabel('train batch idx')
ax2.set_ylabel('MSELoss')
ax2.set_title('Before training energy loss')
ax2.grid()

plt.tight_layout()
plt.savefig('equiv_inv.pdf')