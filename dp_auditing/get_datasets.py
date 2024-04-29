import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, TensorDataset, ConcatDataset
import torch
from opacus import PrivacyEngine
from torch import nn, optim
from torchvision import datasets, transforms
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

def get_dataset(dataset, data_path, batch_size):
    if dataset == 'mnist':
        train_set = datasets.MNIST(data_path, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

        test_set = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        num_classes = 10
        num_features = 28 * 28
        input_shape = (1, 28, 28) 

    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

        num_classes = 10
        num_features = 3 * 32 * 32
        input_shape = (3, 32, 32) 

    else:
        raise ValueError(f"Dataset {dataset} not recognized")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_features, num_classes, input_shape