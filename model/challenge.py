"""
EECS 445 - Introduction to Machine Learning
Winter 2022 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: define each layer of your network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(
            5, 5), stride=(2, 2), padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
            5, 5), stride=(2, 2), padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(
            5, 5), stride=(2, 2), padding=2)
        self.fc_1 = nn.Linear(in_features=128, out_features=2)



        ##

        self.init_weights()

    def init_weights(self):
        # TODO: initialize the parameters for your network
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        # TODO: initialize the parameters for [self.fc1]
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(128))
        nn.init.constant_(self.fc_1.bias, 0.0)

        # nn.init.normal_(self.fc_2.weight, 0.0, 1 / sqrt(32))
        # nn.init.constant_(self.fc_2.bias, 0.0)

        # nn.init.normal_(self.fc_3.weight, 0.0, 1 / sqrt(32))
        # nn.init.constant_(self.fc_3.bias, 0.0)

        ##

    def forward(self, x):
        """ You may optionally use the x.shape variables below to resize/view the size of 
            the input matrix at different points of the forward pass
        """
        N, C, H, W = x.shape

        # TODO: forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128)
        x = self.fc_1(x)

        ##

        return x
