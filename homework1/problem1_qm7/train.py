import os
import pickle
import sys
import numpy as np
import model as nn
import copy
import scipy
import scipy.io
import argparse
from tqdm import tqdm
from utils import rotate_3d_coordinates


def train(nnsgd, nnavg, R, Z, E_true, mb, hist, augment, schedule=False):
    rmses = []
    for i in tqdm(range(1, 100000)):

        # learning rate schedule, changed for speed
        if schedule and i > 25500: lr = 0.00000001  # custom schedule
        elif i > 2500:  lr = 0.0000005
        elif i > 500:   lr = 0.00000025
        elif i > 0:     lr = 0.0000001
        # if i > 12500: lr = 0.000001  #bug in the code

        # sample minibatch indices
        r = np.random.randint(0, len(R), [mb])

        # My code
        input_nn = R[r]
        if augment:
            x_degrees, y_degrees, z_degrees = np.random.uniform(-180, 180, size=(3,))/180*np.pi  # convert to radiant
            input_nn = rotate_3d_coordinates(R[r], x_degrees, y_degrees, z_degrees)

        E_pred = nnsgd.forward((input_nn, Z[r]))  # TODO: fill in

        # I choose a loss that is an array (1,1), instead of a scalar
        #Loss = 0.5*((E_pred - E_true[r])**2).sum(axis=0, keepdims=True)   # TODO: fill in
        # print(f"{loss}")
        dLoss = (E_pred - E_true[r])
        rmse = np.square(E_pred - E_true[r]).mean(axis=0) ** .5

        rmses.append(rmse)
        nnsgd.backward(dLoss)  # TODO: add argument to backward
        nnsgd.update(lr)
        nnavg.average(nnsgd, (1 / hist) / ((1 / hist) + i))
        nnavg.nbiter = i

        if i % 10000 == 0:
            print(i, end= ' ')
            print(f"RMSE: {sum(rmses[-100:]) / 100} kcal/mol")

    if schedule: str_schedule = '_custom_schedule'
    else: str_schedule = ''
    pickle.dump(nnavg, open(f'nn-augment={augment}{str_schedule}.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    np.save(f'training-rmses-augment={augment}{str_schedule}.npy', np.array(rmses))
    print("Done Training")


if __name__ == '__main__':
    '''Setup'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--augmentation', action='store_true', help='whether to perform data augmentations')
    parser.add_argument('--train_fraction', type=float, default=0.5, help='Fraction of training split to use')
    parser.add_argument('--mb', type=int, default=25, help='Batch size')
    parser.add_argument('--hist', type=float, default=0.1,
                        help='Fraction of the history to be remembered for Exponential Moving Average')

    parser.add_argument('--schedule', action='store_true', help='Custom Learning Rate Schedule')

    args = parser.parse_args()

    np.random.seed(args.seed)

    '''Load data'''
    if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
    dataset = scipy.io.loadmat('qm7.mat')
    split_idx = dataset['P'][1:].flatten()  # leave first split for testing
    skip_every = int(1 / args.train_fraction)
    R = dataset['R'][split_idx][::skip_every]
    Z = dataset['Z'][split_idx][::skip_every]
    E_true = dataset['T'][0, split_idx][::skip_every]

    '''Create neural network'''
    # Define input and output layers
    I, O = nn.Input((R, Z)), nn.Output(E_true)
    print(I.nbout, O.nbinp)
    nnsgd = nn.Sequential([I, nn.Linear(I.nbout, 400), nn.Tanh(),
                           nn.Linear(400, 100), nn.Tanh(),
                           nn.Linear(100, O.nbinp), O
                           ])  # TODO: fill in layers.
    nnavg = copy.deepcopy(nnsgd)

    '''Train neural network'''
    train(nnsgd, nnavg, R, Z, E_true, args.mb, args.hist, args.augmentation, args.schedule)