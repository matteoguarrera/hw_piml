import torch
import torch.nn as nn
import numpy as np
from utils import set_seed
from torchdiffeq import odeint

class ModelDirect(nn.Module):
    """output: θ(t + 1), velocity v(t + 1)
       input:  θ(t), velocity v(t)"""
    def __init__(self, cfg):
        super(ModelDirect, self).__init__()

        set_seed(seed=10 ** 3)  # for reproducibility
        self.criterion = cfg.criterion
        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=cfg.hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(in_features=cfg.hidden_dim, out_features=2, dtype=torch.float32)
        ).to(cfg.device)

    def forward(self, x):
        return self.model(x)

    def autoregressive_evolution(self, test_loader: torch.utils.data, ) -> list:
        # Autoregressive steps
        logger_loss = []
        x_t, target = next(iter(test_loader))  # init
        autoregressive_traj = [x_t.squeeze().detach().numpy()]

        with torch.no_grad():
            for i, (_, target) in enumerate(test_loader):
                x_tp1 = self(x_t)
                autoregressive_traj.append(x_tp1.squeeze().detach().numpy())
                loss = self.criterion(target, x_tp1)
                logger_loss.append(loss.item())
                x_t = x_tp1
            traj = np.array(autoregressive_traj).T
            print(traj.shape)
            return traj, logger_loss


class NeuralODE(nn.Module):
    def __init__(self, cfg, solver):
        super(NeuralODE, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=cfg.hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(in_features=cfg.hidden_dim, out_features=2, dtype=torch.float32)
        ).to(cfg.device)
        self.solver = solver
    def forward(self, t, x):
        """forward pass has a dummy input t as requested by odeint"""
        return self.model(x)

    def autoregressive_evolution(self, test_loader: torch.utils.data, grid: torch.Tensor) -> list:
        with torch.inference_mode():
            x, _ = next(iter(test_loader))  # sample, target

            autoregressive_traj = odeint(func=self, y0=x, t=grid, method=self.solver)
            return autoregressive_traj.squeeze().T  # [2, 250]
