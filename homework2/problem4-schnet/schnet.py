import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Optional
import ase
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


'''
pip install ase, lmdb, wandb
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html

'''
class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 5.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None):
        super().__init__()

        

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None

        self.atomic_mass = torch.from_numpy(ase.data.atomic_masses)

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.reset_parameters()


    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, z, pos, batch=None):
        """"""
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch
        pos.requires_grad = True

        print(z.shape, pos.shape)

        # TO DO: compute initial node representations
        h = self.embedding(z)  # Correct

        # Computes graph edges to all points within a given distance.
        # DOC: https://pytorch-geometric.readthedocs.io/en/1.3.1/modules/nn.html#torch_geometric.nn.pool.radius_graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        # edge_index <= return torch.stack([row, col], dim=0)   [torch.Long]

        # TO DO: compute pairwise distances
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)

        # TO DO: compute radial basis expansion of pairwise distances
        edge_attr = self.distance_expansion(edge_weight)  # Correct

        for interaction in self.interactions:
            # TODO: call interaction layers
            # InteractionBlock, Figure 2 from the paper
            h = h + interaction(h, edge_index, edge_weight, edge_attr)


        #linear projection to obtain atomwise energies
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # TODO: compute global energies per molecule
        """ Why all of this? 
        Just because it's there in the original codebase? """
        if self.dipole:
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            # scale energy contributions in the original codebase
            h = h * self.std + self.mean

        energy = scatter(h, batch, dim=0, reduce=self.readout)

        # TODO: compute forces
        forces = -1 * torch.autograd.grad(energy,
                                          pos,
                                          grad_outputs=torch.ones_like(energy),
                                          create_graph=True,
                                          retain_graph=True)[0]
        # forces = -energy.grad

        return energy, forces


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        # TO DO: fill in this line
        h = self.conv(h, edge_index, edge_weight, edge_attr)   # Correct
        h = self.act(h)
        h = self.lin(h)
        return h


class CFConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__()
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        """ From the paper:
        1) RBF,                  2) Dense,
        3) Shifted Softplus,     4) Dense,
        5) Shifted Softplus,     6) Out """

        print('h ', h.shape,'edge idx ', edge_index.shape,
              'edge wei ',  edge_weight.shape, 'edge attr ', edge_attr.shape)

        h = self.lin1(h)

        # TO DO: transform edge weights to [0,1], aka use the equation with cosine
        c = 0.5 * (torch.cos(edge_weight *torch.pi/self.cutoff) +1.0)  # Correct
        print('c ', c.shape)

        # TO DO: compute filters
        W = self.nn(edge_attr)* c * h  # W_theta in the text of the HW
        print('W ', W.shape)

        # TODO: perform graph convolution
        # DOC: https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        h = scatter(W, edge_index, dim=1, reduce="sum")
        h = self.lin2(h)
        return h
        


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)  # positions of the centers
        self.gamma = 0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset) #sets the attribute self.offset = offset

    def forward(self, dist):
        #TODO: implement this function
        out = torch.exp(-self.gamma*(dist-self.offset)**2)
        # assert
        print(dist.shape, self.offset.shape, self.gamma.shape, out.shape)
        return out

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
