# Physics-Inspired Machine Learning Assignment 1

We recommend using anaconda to manage python environments. If you're using anaconda:

```
conda create -n physics-ml python==3.8
conda activate physics-ml
pip install -r requirements.txt
```

This will create a new conda environment called ```physics-ml``` with all the required packages. 
Otherwise, you may use pip directly:

```pip install -r requirements.txt```

For any questions, please make a piazza post.

## Note on CUDA versions
PyTorch's default installation is CUDA 12.1. If your GPU is CUDA 11.8, use the below pip command to install PyTorch:

```pip install torch --index-url https://download.pytorch.org/whl/cu118```
