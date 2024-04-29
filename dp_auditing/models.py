import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, TensorDataset, ConcatDataset
import torch
from opacus import PrivacyEngine
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import numpy as np
import os
import random

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pickle
import json
import numpy as np
import os
from collections import defaultdict
import pandas as pd
from urllib.request import urlretrieve

import string

import warnings

from joblib import Parallel, delayed
from copy import deepcopy
import matplotlib.pyplot as plt

import random
import torch

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='opacus')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='opacus')

# Suppress specific UserWarning by message
warnings.filterwarnings("ignore", message="Secure RNG turned off.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha.*", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in log", category=RuntimeWarning)
# Suppress the specific deprecation warning about non-full backward hooks
warnings.filterwarnings("ignore", message="Using a non-full backward hook.*", category=UserWarning)


class MNISTNet(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

class CIFAR10Model(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.layer_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(32, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(64, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Conv2d(256, 10, (3, 3), padding=1, stride=(1, 1)),
        ])

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
            # print(x.shape)
        return torch.mean(x, dim=(2, 3))
    

def RESNET16_model(num_classes=10):
    """
    Create a ResNet50 model instance with the final layer adjusted for CIFAR10.
    """
    # model = models.resnet50(pretrained=False)
    # model = models.wide_resnet50_2(pretrained=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = wide_resnet16_4()
    return model

###################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResNet(nn.Module):
    def __init__(self, block, num_blocks, widen_factor=4, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # Adjusting the layers according to the new depth and widen factor
        self.layer1 = self._make_layer(block, 16 * widen_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * widen_factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * widen_factor, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * widen_factor * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def wide_resnet16_4():
    # For WRN-16-4, we use a configuration that adds up to 16 layers in total considering the first conv layer, 3 groups of conv layers, and the final dense layer
    # num_blocks per group to make the total depth to 16: 1 conv layer + (2*3) blocks + 1 final layer = 16 layers
    return WideResNet(BasicBlock, [2, 2, 2], widen_factor=4)

