import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet2D(nn.Module):

    # You may include additional arguments if you wish.
    def __init__(self):
        super(ConvNet2D, self).__init__()
        input_channels = 3
        output_channels = 101*101  # Nx x Nt 101 x 101
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pointwise1 = nn.Linear(652864, 128)
        self.pointwise2 = nn.Linear(128, output_channels)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.pointwise1(x))
        x = self.pointwise2(x)

        return x.view(x.shape[0], 101, 101)