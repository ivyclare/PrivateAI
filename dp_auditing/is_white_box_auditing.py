import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from opacus import PrivacyEngine
from torch import nn, optim
import torch.nn.functional as F
from opacus.validators import ModuleValidator

import numpy as np
import os

import pickle
import json
import pandas as pd
from urllib.request import urlretrieve

import string

import warnings

from copy import deepcopy
import matplotlib.pyplot as plt

import random
import argparse


from get_datasets import get_dataset
from models import MNISTNet, wide_resnet16_4
from train import train_model, calculate_accuracy, evaluate_model, train_model_with_memory_manager
from utils import create_canary_samples

import sys
 
# setting path
sys.path.append('../immediate_sensitivity_experiments')
from calculate_immediate_sensitivity import calculate_immediate_sensitivity


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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
import numpy as np
from copy import deepcopy
from torchvision import datasets, transforms


def compute_input_canary_gradient(model, criterion, device, input_shape, clipping_value, num_classes=10):
    """
    Computes gradients for a blank input to the model, applies clipping, and returns the gradients directly.
    """
    model.zero_grad()
    blank_input = torch.zeros([1, *input_shape], device=device)
    target = torch.randint(0, num_classes, (1,), device=device)
    output = model(blank_input)
    loss = criterion(output, target)
    loss.backward()
    
    gradients = [param.grad.clone() for param in model.parameters() if param.grad is not None]
    clipped_gradients = [torch.clamp(grad, -clipping_value, clipping_value) for grad in gradients]
    
    return clipped_gradients

def create_canary_gradient(shape, clipping_value):
    """
    Creates a canary gradient with zeros everywhere except for a single index set to the clipping value.
    """
    canary_grad = torch.zeros(shape)
    canary_grad.view(-1)[0] = clipping_value  #choose the first index for simplicity
    return canary_grad

def dot_product(gradient1, gradient2):
    """
    Compute dot product between two gradient vectors.
    """
    return (gradient1 * gradient2).sum().item()
            
def train_model_with_is(alpha, epsilon, model_base, loader_base, model_canary, loader_canary, criterion, optimizer_base, optimizer_canary, epochs=1, device='cpu', clipping_value=1, input_shape=(3, 32, 32), gradient_canary=False, input_canary=False, qc=1, target_observations=5000):

    observations, observations_canary = [], []
    observation_num = 0 
    
    model_base.train()
    model_canary.train()

    #epsilon_iter = epsilon / (epochs * loader_base.batch_size)

    for epoch in range(epochs):
        if observation_num >= target_observations:
            break  # Stop training if target number of observations has been reached

        total_loss, total_correct, total_samples = 0, 0, 0

        canary_iter = iter(loader_canary)    # Assume loader_canary is an iterable that can be restarted each epoch

        print("\nTraining  MODEL")
        # BASE Model Training
        sigmas = []
        sigmas_canary = []
        
        for data, targets in loader_base:
            data, targets = data.to(device), targets.to(device)
            optimizer_base.zero_grad()

            loss, batch_sensitivities, output = calculate_immediate_sensitivity(model_base, criterion, data, targets)
            
            batch_sensitivity = np.max(batch_sensitivities) / loader_base.batch_size

            sigma = np.sqrt((batch_sensitivity**2 * alpha) / (2 * epsilon/(epochs * loader_base.batch_size)))
            sigmas.append(sigma)
            with torch.no_grad():
                for p in model_base.parameters():
                    p.grad += (sigma * torch.randn(1).to(device))


            optimizer_base.step()
            
            # Create Canary gradients 
            canary_grad = None
            if gradient_canary:
                canary_grad = create_canary_gradient(next(model_base.parameters()).shape, clipping_value).to(device)
            elif input_canary:
                canary_grad = compute_input_canary_gradient(model_base, criterion, device, input_shape, clipping_value) #.to(device)


            # Extract gradients for observation
            grad_base = next(model_base.parameters()).grad.view(-1)

            # CANARY Model Training
            try:
                data_0, targets_0 = next(canary_iter)
            except StopIteration:                       # Restart the canary iterator if it runs out of data
                canary_iter = iter(loader_canary)
                data_0, targets_0 = next(canary_iter)

            data_0, targets_0 = data_0.to(device), targets_0.to(device)  # Sampled batches for canary model
            optimizer_canary.zero_grad()   #

            loss_canary, batch_sensitivities_canary, output_canary = calculate_immediate_sensitivity(model_canary, criterion, data_0, targets_0)

            batch_sensitivity_canary = np.max(batch_sensitivities_canary) / loader_canary.batch_size

            sigma_canary = np.sqrt((batch_sensitivity_canary**2 * alpha) / (2 * epsilon/(epochs * loader_canary.batch_size)))
            sigmas_canary.append(sigma_canary)
            with torch.no_grad():
                for p in model_canary.parameters():
                    p.grad += (sigma_canary * torch.randn(1).to(device))

            optimizer_canary.step()
            
            # # Apply canary gradients with probability qc to model_canary
            if np.random.rand() < qc:
                if gradient_canary and canary_grad is not None:     # Gradient Canary
                    for param in model_canary.parameters():
                        if param.grad is not None:
                            param.grad += canary_grad.to(param.device)  
                            break  
                    
                    grad_base = next(model_base.parameters()).grad.view(-1)
                    grad_canary = next(model_canary.parameters()).grad.view(-1)

                    observation_base = dot_product(canary_grad.view(-1), grad_base)
                    observation_canary = dot_product(canary_grad.view(-1), grad_canary)

                elif input_canary and canary_grad is not None:       # Input Canary
                    for param, canary_grad_tensor in zip(model_canary.parameters(), canary_grad):
                        if param.grad is not None:
                            param.grad += canary_grad_tensor.to(param.device)
                    canary_grad = torch.cat([g.view(-1) for g in canary_grad])

                    grad_base_flat = torch.cat([g.grad.view(-1) for g in model_base.parameters() if g.grad is not None])
                    grad_canary_flat = torch.cat([g.grad.view(-1) for g in model_canary.parameters() if g.grad is not None])
                    
                    # # Compute observations using the prepared gradients
                    observation_base = dot_product(canary_grad, grad_base_flat)
                    observation_canary = dot_product(canary_grad, grad_canary_flat)

            observations.append(observation_base)
            observations_canary.append(observation_canary)
            observation_num += 1 

            if observation_num >= target_observations:
                break  # Stop training if target number of observations has been reached

            
            # Save the observations after each observation_num to the file
            if gradient_canary:
                if not os.path.exists('./cifar_grad_observations/'):
                    os.makedirs('./cifar_grad_observations/')
                filename = f'./cifar_grad_observations/observations_gradient_canary.pkl'
            else:
                if not os.path.exists('./cifar_input_observations/'):
                    os.makedirs('./cifar_input_observations/')
                filename = f'./cifar_input_observations/observations_input_canary.pkl'
            with open(filename, 'wb') as f:
                pickle.dump((observations, observations_canary), f)

            print(f"Saved observations to {filename} after processing observation number {observation_num}")

                
            # Replace the weights of the canary model with the base model
            model_canary.load_state_dict(model_base.state_dict())

            # accuracy calculation
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item()

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(loader_base)
        avg_accuracy = total_correct / total_samples
        # epsilon = privacy_engine_base.accountant.get_epsilon(delta=1e-5)
        
        print(f'Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, Epsilon: {epsilon:.2f}')


def white_box_auditing(dataloader, model_function, target_observations, epochs_per_observation, lr, clipping_value, target_epsilon, target_delta, device='cpu', qc=1, gradient_canary=True, input_canary=False, input_shape=(1, 28, 28)):
    criterion = nn.CrossEntropyLoss()

    # for observation_num in range(target_observations):
    model_base = model_function().to(device)
    model_base = ModuleValidator.fix(model_base)
    ModuleValidator.validate(model_base, strict=False)

    model_canary = deepcopy(model_base)
    model_canary = ModuleValidator.fix(model_canary)
    ModuleValidator.validate(model_canary, strict=False)

    optimizer_base = optim.SGD(model_base.parameters(), lr=lr)
    optimizer_canary = optim.SGD(model_canary.parameters(), lr=lr)

    loader_base = dataloader
    loader_canary = dataloader

    # privacy_engine_base = PrivacyEngine(accountant='rdp')
    # model_base, optimizer_base, loader_base = privacy_engine_base.make_private_with_epsilon(
    #     module= model_base,
    #     optimizer=optimizer_base,
    #     data_loader=dataloader,                     # FIX: Create both dataloaders from the same dataloader
    #     epochs=epochs_per_observation,
    #     target_epsilon=target_epsilon,
    #     target_delta=target_delta,
    #     max_grad_norm=clipping_value
    # )
    
    # privacy_engine_canary = PrivacyEngine(accountant='rdp')
    # model_canary, optimizer_canary, loader_canary = privacy_engine_canary.make_private_with_epsilon(
    #     module=model_canary,
    #     optimizer=optimizer_canary,
    #     data_loader=dataloader,                 # FIX: Create both dataloaders from the same dataloader
    #     epochs=epochs_per_observation,
    #     target_epsilon=target_epsilon,
    #     target_delta=target_delta,
    #     max_grad_norm=clipping_value
    # )

    train_model_with_is(2, .1, model_base, loader_base, model_canary, loader_canary, criterion, optimizer_base, optimizer_canary, epochs_per_observation, device, clipping_value, input_shape, gradient_canary, input_canary, qc, target_observations)

    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Run white-box privacy auditing with canary samples.')
    parser.add_argument('--data_path', type=str, default='.', help='Path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Name of the dataset')
    parser.add_argument('--num_observations', type=int, default=5, help='Number of observations to collect')
    parser.add_argument('--epochs_per_observation', type=int, default=1, help='Number of epochs per observation')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=float, default=64, help='Batch size')
    parser.add_argument('--target_epsilon', type=float, default=3.0, help='DP epsilon')
    parser.add_argument('--target_delta', type=float, default=1e-5, help='DP delta')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--file_number', type=int, default=1, help='File number for saving observations')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[1, 28, 28], help='Input shape of the model')
    parser.add_argument('--gradient_canary', action='store_true', help='Use gradient canary')
    parser.add_argument('--input_canary', action='store_true', help='Use input canary')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, _, _, input_shape = get_dataset(args.dataset_name, args.data_path, args.batch_size)

    def model_function():
        model = wide_resnet16_4() 
        model.to(device)
        return model

    white_box_auditing(
        dataloader=train_loader, #.dataset,
        model_function=model_function,
        # model_function=lambda: RESNET16_model(num_classes=10), 
        target_observations=args.num_observations,
        epochs_per_observation=args.epochs_per_observation,
        lr=args.lr,
        clipping_value=args.max_grad_norm,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        device=device,
        qc=1,  # Assuming qc is always 1 for this setup
        gradient_canary=args.gradient_canary,
        input_canary=args.input_canary,
        input_shape=input_shape, #tuple(args.input_shape),
    )

if __name__ == "__main__":
    main()