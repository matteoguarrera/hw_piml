# Physics-Inspired Machine Learning - Assignment 2

## Installation and Requirements
In this assignment, you'll be using a variety of PyTorch extensions including pytorch\_geometric, pytorch\_scatter, and pytorch\_cluster. We have found that installing all the required packages to be troublesome, so we have provided a docker container which will automatically build an environment for you. Alternatively, you are free to try and install all of the required packages yourself. Please post any questions to Piazza.

### Self-installation with conda

```bash
# Install PyTorch version 2.1.0 - You MUST use this version
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --user pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install --user torch_geometric
pip install --user ase wandb lmdb
```

### Docker Container (Recommended)
To use the docker container, you must be on a Linux based operating system and have an Nvidia GPU. 

**(0). Install pre-requisites.** For your distribution, install ```curl```. E.g. In Ubuntu: ```sudo apt install curl```.

**(1) Install Docker.**
Follow the instructions on: https://docs.docker.com/engine/install/

**(2) Install nvidia-container-toolkit.**
Redhat based distributions:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Debian based distributions:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

If you are unsure of distribution you have, run either ```which apt```. If there is no output, you have a Redhat distribution; otherwise you have a Debian based distribution.

**(3) Building the container and running commands.**

To build the container, run ```python do.py build```. You only need to do this once.

To run a bash command in the container (e.g., ```python train.py```) use:

```bash
python do.py run --bash='$CMD'
```

To open an interactive terminal in the container, run ```python do.py run```.

**(4) Removing hanging containers.**

All running containers should exit and cleanup on their own when you finish. If they do not, run ```python do.py stop```.

