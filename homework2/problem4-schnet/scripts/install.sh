#!/bin/bash --login

# Install pip and upgrade it
conda install -y pip
pip install --upgrade pip

# Install jupyterlab
pip install --no-cache-dir jupyterlab

# Install some modules for jupyterlab
pip install --user ipywidgets

########################################################################################################################################################################
# Install python packages required for the homework here
########################################################################################################################################################################
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install --user pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install --user torch_geometric
pip install --user ase
pip install --user lmdb wandb
########################################################################################################################################################################
