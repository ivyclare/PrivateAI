
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
import torch

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='opacus')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='opacus')

# Suppress specific UserWarning by message
warnings.filterwarnings("ignore", message="Secure RNG turned off.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha.*", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in log", category=RuntimeWarning)

# # Suppress all UserWarnings and RuntimeWarnings (use with caution)
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress the specific deprecation warning about non-full backward hooks
warnings.filterwarnings("ignore", message="Using a non-full backward hook.*", category=UserWarning)



import torch
import random

def create_canary_samples(data_loader, num_samples, mislabeled=False, random_label=False):
    for batch in data_loader:
        data_shape = batch[0].shape[1:]  # Exclude batch dimension
        break  # Only need the first batch to determine shape
    
    # Ensure num_samples is an integer
    num_samples = int(num_samples)
    
    # Convert data_shape elements to integers if necessary (shouldn't be needed but just in case)
    data_shape = tuple(int(dim) for dim in data_shape)
    
    if mislabeled:
        all_data, all_labels = [], []
        for data, labels in data_loader:
            all_data.append(data)
            all_labels.append(labels)

        all_data = torch.cat(all_data)[:num_samples]
        all_labels = torch.cat(all_labels)[:num_samples]

        if random_label:
            unique_labels = list(range(10))  # Assuming MNIST with labels 0-9
            new_labels = [random.choice([l for l in unique_labels if l != label.item()]) for label in all_labels]
            all_labels = torch.tensor(new_labels, dtype=all_labels.dtype)
    else:
        # Create zero-valued tensors for data and labels
        all_data = torch.zeros((num_samples,) + data_shape)
        all_labels = torch.zeros(num_samples, dtype=torch.long)

    return all_data, all_labels


# python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 5000 --epochs_per_observation 2 --lr 0.01 --target_epsilon 8 --file_number=2 --input_canary
# python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 5000 --epochs_per_observation 2 --lr 0.01 --target_epsilon 8 --file_number=1 --gradient_canary

# python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 2 --epochs_per_observation 2 --lr 0.01 --target_epsilon 8 --file_number=2 --input_canary
# python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 2 --epochs_per_observation 2 --lr 0.01 --target_epsilon 8 --file_number=1 --gradient_canary


# python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 5000 --epochs_per_observation 80 --lr 0.01 --target_epsilon 8 --input_canary
# python dp_white_box_auditing.py --dataset_name "cifar10" --num_observations 5000 --epochs_per_observation 80 --lr 0.01 --target_epsilon 8 --gradient_canary

###
# python -u dp_feature_space.py --dataset='cifar10' --num_epochs=20 --lr=0.01 --batch_size=64 --target_epsilon=8 
