import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from math import ceil
import numpy as np
from scipy.special import ellipj, ellipk
import matplotlib.pyplot as plt
import torchdiffeq
from copy import deepcopy
from types import SimpleNamespace
import scipy as sci
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint


def solution_pendulum_theta(grid, theta0):
    '''
    Analytical solution for the undamped, non-linear pendulum using elliptical integrals
    grid: array of time points of shape [traj_length]
    theta0: initial angular displacement in radians (float)
    '''

    S = np.sin(0.5*(theta0))
    K_S = ellipk(S**2)
    omega_0 = np.sqrt(9.81)
    sn,cn,dn,ph = ellipj( K_S - omega_0*grid, S**2 )
    theta = 2.0*np.arcsin( S*sn)
    d_sn_du = cn*dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
    return np.stack([theta, d_theta_dt], axis=0)


def set_seed(seed=10**3):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def visualize_pendulum_data(grid, x_train, x_test):
    '''
    grid: array of time points of shape [traj_length]
    x_train: training trajectory of shape [2 x traj_length]
    x_test: test trajectory of shape [2 x traj_length]
    '''
    ### Plot data
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(121)
    ax.plot(grid, x_train[0], 'o--', label=r'$\theta(0)=2.6$', c='#dd1c77')
    ax.plot(grid, x_test[0],'s--', label=r'$\theta(0)=1.4$', c='#2c7fb8')
    ax.tick_params(axis='x', labelsize=14) 
    ax.tick_params(axis='y', labelsize=14) 
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylabel(r'angular displacement, ${\theta}(t)$', fontsize=14)
    ax.set_xlabel('time, t', fontsize=14)
    ax.legend(loc='lower right', fontsize=14)

    ax2 = fig.add_subplot(122)
    ax2.plot(grid, x_train[1], 'o--', label=r'$\theta(0)=2.6$', c='#dd1c77')
    ax2.plot(grid, x_test[1], 's--', label=r'$\theta(0)=1.4$', c='#2c7fb8')
    ax2.tick_params(axis='x', labelsize=14) 
    ax2.tick_params(axis='y', labelsize=14) 
    ax2.tick_params(axis='both', which='minor', labelsize=14)
    ax2.set_ylabel(r'angular velocity, ${v}(t)$', fontsize=14)
    ax2.set_xlabel('time, t', fontsize=14)
    ax2.legend(loc='lower right', fontsize=14)
    plt.savefig("pendulum_trajectories.png")


def visualize_predictions(grid, x_test, x_preds_direct, x_preds_odenet, x_preds_odenet_rk4, time_horizon = 1, dt = 0.1):
    '''
    x_test: ground truth test trajectory of shape [2 x traj_length]
    x_preds_direct: direct prediction trajectory of shape [2 x traj_length]
    x_preds_odenet: neural ODE prediction trajectory of shape [2 x traj_length]
    time_horizon: horizon of prediction during training
    dt: timestep used for test-time prediction
    '''
    dt_ratio = ceil(dt / 0.1)
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(121)
    ax.plot(grid[::dt_ratio], x_test[0, ::dt_ratio], 'o--', label=r'truth', c='#2c7fb8')
    ax.plot(grid[::dt_ratio], x_preds_direct[0, ::dt_ratio], 's--', label=r'Direct Net', c='#de2d26')
    ax.plot(grid[::dt_ratio], x_preds_odenet[0], 's--', label=r'ODENet', c='#2ca25f')
    ax.plot(grid[::dt_ratio], x_preds_odenet_rk4[0], 's--', label=r'ODENet_RK4', c = '#eda323')


    ax.tick_params(axis='x', labelsize=14) 
    ax.tick_params(axis='y', labelsize=14) 
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylabel(r'angular displacement, ${\theta}(t)$', fontsize=14)
    ax.set_xlabel('time, t', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)

    ax2 = fig.add_subplot(122)
    ax2.plot(grid[::dt_ratio], x_test[1, ::dt_ratio], 'o--', label=r'truth', c='#2c7fb8')
    ax2.plot(grid[::dt_ratio], x_preds_direct[1, ::dt_ratio], 's--', label=r'Direct Net', c='#de2d26')
    ax2.plot(grid[::dt_ratio], x_preds_odenet[1], 's--', label=r'ODENet', c='#2ca25f')
    ax2.plot(grid[::dt_ratio], x_preds_odenet_rk4[1], 's--', label=r'ODENet_RK4', c = '#eda323')


    ax2.tick_params(axis='x', labelsize=14) 
    ax2.tick_params(axis='y', labelsize=14) 
    ax2.tick_params(axis='both', which='minor', labelsize=14)
    ax2.set_ylabel(r'angular velocity, ${v}(t)$', fontsize=14)
    ax2.set_xlabel('time, t', fontsize=14)    
    plt.savefig(f"plots/predicted_pendulum_trajectories_timehorizon={time_horizon}_dt={dt}.png")


