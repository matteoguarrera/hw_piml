import comet_ml
import torch
import argparse
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import copy
torch.manual_seed(0)

from losses import rel_data_loss,  darcy_pde_residual
from data import DarcyFlowDataset
from model import ResNet, ConstrainedModel

def eval(model, val_loader, exp: comet_ml.Experiment, epoch = 0):
    running_val_data_loss = []
    model = copy.deepcopy(model)
    for mesh, diffusion_coeff, solution in tqdm(val_loader, desc=f'Val Epoch {epoch}'):
        # Forward pass
        pred = model(mesh, diffusion_coeff)
        # Loss
        data_loss = rel_data_loss(pred, solution)
        pde_loss = darcy_pde_residual(pred, mesh, diffusion_coeff)
        running_val_data_loss.append(data_loss.item())
    avg_val_data_loss = sum(running_val_data_loss) / len(running_val_data_loss)
    exp.log_metrics({
        'val_data_loss': avg_val_data_loss
    }, epoch=epoch)
    exp.flush()

def train(model: torch.nn.Module, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer, 
            constrained: bool,
            epochs: int,
            exp: comet_ml.Experiment):
    
    with torch.no_grad():
        rng_state = torch.get_rng_state()
        eval(model, val_loader, exp)
        torch.set_rng_state(rng_state)


    for epoch in range(epochs):
        model.train()
        for mesh, diffusion_coeff, solution in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            # Forward pass
            pred = model(mesh, diffusion_coeff)
            # Loss
            data_loss = rel_data_loss(pred, solution)
            pde_loss = darcy_pde_residual(pred, mesh, diffusion_coeff)
            loss = pde_loss
            # Backward pass
            loss.backward()
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
            optimizer.step()

            exp.log_metrics({
                'train_data_loss': data_loss.item(),
                'train_pde_loss': pde_loss.item(),
                'train_loss': loss.item(),
                'train_grad_norm': norm.item()
            })

        with torch.no_grad():
            eval(model, val_loader, exp, epoch=epoch+1)
        if constrained:
            for g in optimizer.param_groups:
                g['lr'] *= .1

        exp.flush()
    

def main():
    # TODO - Enable logging
    COMET_ML_WORKSPACE = ''
    COMET_ML_API_KEY = ''

    parser = argparse.ArgumentParser('Train the 2D Darcy Flow model.')
    parser.add_argument('--data_dir', type=str, help='Path to the data file.', default='data/')

    # Hyper parameters
    parser.add_argument('--constrained', help='Whether to use the constrained model.', action='store_true')
    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer.', default=1e-3)
    parser.add_argument('--batch_size', type=int, help='Batch size for training.', default=8)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model.', default=5)
    parser.add_argument('--num_basis_functions', type=int, help='Number of basis functions for the constrained model.', default=4000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloaders
    train_dataset = DarcyFlowDataset(os.path.join(args.data_dir, 'piececonst_r241_N1024_smooth1.mat'), device)
    val_dataset = DarcyFlowDataset(os.path.join(args.data_dir, 'piececonst_r241_N1024_smooth2.mat'), device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    if not args.constrained:
        model = ResNet(in_dim=3, hidden_dims=[64, 64, 64], out_dim=1).to(device)
    else:
        model = ConstrainedModel(in_dim=3, hidden_dims=[64, 64, 64], 
                                 n_basis_functions=args.num_basis_functions).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    exp = comet_ml.Experiment(project_name="pi-ml-assignment3-darcy-flow",
                            workspace=COMET_ML_WORKSPACE,
                            api_key=COMET_ML_API_KEY)
    exp.log_parameters({
        'lr': args.lr,
        'batch_size': args.batch_size,
        'constrained': args.constrained,
    })


    train(model, train_loader, val_loader, optimizer, constrained=args.constrained,
        epochs=args.epochs,
        exp=exp)
    
    exp.end()



if __name__ == '__main__':
    main()
