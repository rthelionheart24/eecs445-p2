"""
EECS 445 - Introduction to Machine Learning
Winter 2022 - Project 2
Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Source(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        self.conv1 =
        self.pool =
        self.conv2 =
        self.conv3 =
        self.fc1 =
        ##

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]

        ##

    def forward(self, x):
        """ You may optionally use the x.shape variables below to resize/view the size of
            the input matrix at different points of the forward pass
        """
        N, C, H, W = x.shape

        ## TODO: forward pass

        ##

        return z
