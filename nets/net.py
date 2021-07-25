import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim


class SDF_Net(nn.Module):
    def __init__(self):
        super(SDF_Net, self).__init__()
        # TODO: Network structure

    def forward(self, x):
        x = torch.Tensor(x)
        # TODO: Network structure
        return x


if __name__ == 'main':
    pass
