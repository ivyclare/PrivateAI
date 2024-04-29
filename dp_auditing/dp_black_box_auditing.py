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
from opacus.validators import ModuleValidator

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
import argparse


from get_datasets import get_dataset
from models import MNISTNet, RESNET50_model
from train import train_model, calculate_accuracy, evaluate_model, train_model_with_memory_manager
from utils import create_canary_samples


import warnings
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


def evaluate_model_and_compute_log_odds(model, sample_data, sample_label, device):
    model.eval()  
    sample_data = sample_data.to(device)
    sample_label = sample_label.to(device).long()
    with torch.no_grad():
        outputs = model(sample_data)
        probabilities = F.softmax(outputs, dim=1)
        true_class_probability = probabilities.gather(1, sample_label.view(-1, 1)).squeeze()
        
        # For multi-class, compute the log odds as log(p/ (1 - p))
        other_class_probabilities = 1 - true_class_probability
        log_odds = torch.log(true_class_probability / other_class_probabilities)
        return log_odds.item()

def train_and_evaluate_models(model_class, train_loader, test_loader, criterion, sample, num_epochs, learning_rate, target_epsilon, target_delta, max_grad_norm, device, sample_num, is_canary=False):
    # Train on Dataset D (original dataset)
    base_model = model_class().to(device)
    base_model = ModuleValidator.fix(base_model)
    ModuleValidator.validate(base_model, strict=False)
    base_optimizer = optim.SGD(base_model.parameters(), lr=learning_rate)
    base_model, _ = train_model(base_model, train_loader, criterion, base_optimizer, num_epochs, target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm, device=device, sample_num=sample_num, is_canary=False)
    # print("Memory Manager")
    # base_model, _ = train_model_with_memory_manager(base_model, train_loader, criterion, base_optimizer, num_epochs, target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm, device=device, sample_num=sample_num, is_canary=False)

    base_model_test_accuracy, base_model_avg_loss = evaluate_model(base_model, test_loader, criterion, device)
    print(f"\nBaseModel Test Acc:{base_model_test_accuracy} Loss:{base_model_avg_loss}")

    # Train on Dataset D' (original dataset combined with the canary sample)
    sample_data, sample_label = sample
    sample_data = sample_data.unsqueeze(0)
    sample_label = sample_label.unsqueeze(0)
    extended_train_loader = DataLoader(ConcatDataset([train_loader.dataset, TensorDataset(sample_data, sample_label)]), batch_size=train_loader.batch_size, shuffle=True)

    canary_model = model_class().to(device)
    canary_model = ModuleValidator.fix(canary_model)
    ModuleValidator.validate(canary_model, strict=False)
    canary_optimizer = optim.SGD(canary_model.parameters(), lr=learning_rate)
    canary_model, _ = train_model(canary_model, extended_train_loader, criterion, canary_optimizer, num_epochs, target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm, device=device, sample_num=sample_num, is_canary=True)
    # canary_model, _ = train_model_with_memory_manager(canary_model, extended_train_loader, criterion, canary_optimizer, num_epochs, target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm, device=device, sample_num=sample_num, is_canary=True)

    canary_model_test_accuracy, canary_model_avg_loss = evaluate_model(canary_model, test_loader, criterion, device)
    print(f"\nCanaryModel Test Acc:{canary_model_test_accuracy} Loss:{canary_model_avg_loss}")

    # Evaluate models and compute log odds
    base_log_odds = evaluate_model_and_compute_log_odds(base_model, sample_data, sample_label, device)
    canary_log_odds = evaluate_model_and_compute_log_odds(canary_model, sample_data, sample_label, device)

    return base_log_odds, canary_log_odds


def black_box_auditing(model_class, criterion, train_loader, test_loader, canary_samples, num_features, num_epochs, learning_rate, target_epsilon, target_delta, max_grad_norm, device, file_number):
    base_observations = []
    canary_observations = []

    for i, sample in enumerate(canary_samples):
        print("---" * 40)
        print("\nProcessing Canary Sample: ", i)
        base_loss, canary_loss = train_and_evaluate_models(
            model_class, train_loader, test_loader, criterion, sample, num_epochs, learning_rate,
            target_epsilon, target_delta, max_grad_norm, device, i
        )

        base_observations.append(base_loss)
        canary_observations.append(canary_loss)
        
        # filename = f'./cifar_observations/observations_{file_number}.pkl'
        filename = f'./cifar_observations/observations_{file_number}.pkl'
        # filename = f'./resultss/observations_{file_number}.pkl'
        # filename = f'./mislabelled_results/observations_{file_number}.pkl'
        # filename = f'observations_{file_number}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump((base_observations, canary_observations), f)
        print(f"Saved observations to {filename} after processing sample {i}")


def main():
    parser = argparse.ArgumentParser(description='Run privacy auditing with canary samples on MNIST dataset.')
    parser.add_argument('--data_path', type=str, default='.', help='Path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--is_dp', type=bool, default=True, help='Differential privacy flag')
    parser.add_argument('--target_epsilon', type=float, default=3.0, help='DP epsilon')
    parser.add_argument('--target_delta', type=float, default=1e-5, help='DP delta')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--num_canary_samples', type=float, default=10, help='Maximum gradient norm for clipping')
    parser.add_argument('--file_number', type=int, required=True, help='File number for saving observations')


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("FILE NUM: ", args.file_number)
    print("Device: ", device)
    print(f"Epochs: {args.num_epochs} , Batch Size: {args.batch_size} , LR: {args.lr}")

    train_loader, test_loader, num_features, num_classes, input_shape = get_dataset(args.dataset_name, args.data_path, args.batch_size)
    
    # model = MNISTNet().to(device)
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    num_canary_samples = int(args.num_canary_samples)
    canary_data, canary_labels = create_canary_samples(train_loader, num_canary_samples, mislabeled=False, random_label=False)
    # canary_data, canary_labels = create_canary_samples(train_loader, num_canary_samples, mislabeled=True, random_label=True)
    # Prepare canary samples for auditing
    canary_samples = [(canary_data[i], canary_labels[i]) for i in range(num_canary_samples)]

    print(canary_data.shape, canary_labels.shape)

    black_box_auditing(
            # MNISTNet,     
            lambda: RESNET50_model(num_classes=10),       
            criterion,           
            train_loader,
            test_loader,       
            canary_samples,      
            num_features,        
            args.num_epochs,     
            args.lr,             
            args.target_epsilon, 
            args.target_delta, 
            args.max_grad_norm,
            device,               
            args.file_number
        )

if __name__ == "__main__":
    main()
