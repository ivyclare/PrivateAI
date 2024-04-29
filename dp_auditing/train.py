import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, TensorDataset, ConcatDataset
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
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
from datetime import datetime


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

import numpy as np
import torch
from opacus import PrivacyEngine



def train_dp_model(model, dataloader, criterion, optimizer, num_epochs, model_name="model", is_dp=True, task_type='multiclass', target_epsilon=1,target_delta=1e-5, max_grad_norm=1, device='cpu'): #, sample_num=1, is_canary=False):
    model.to(device)  
    best_accuracy = 0.0
    best_loss = 0
    best_model_path = None
    model.train()

    if is_dp:
        # delta = 1 / len(dataloader.dataset)
        privacy_engine = PrivacyEngine(accountant='rdp') #accountant=accountant)
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon, 
            target_delta=target_delta,  
            epochs=num_epochs
        )
        print(f"\nUsing epsilon={target_epsilon}, sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        epsilon = 0.0

        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            data = data.float()
            labels = labels.float().view(-1, 1) if task_type == 'binary' else labels.long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(output, labels, task_type=task_type)
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = running_accuracy / len(dataloader)
        print("epoch_accuracy: ", epoch_accuracy)
        if is_dp:
            epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}, Epsilon: {epsilon:.3f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_loss = epoch_loss
            best_model = model
            best_epsilon = epsilon if is_dp else 0
            
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # best_model_path = f"{model_name}_model_epsilon_{best_epsilon:.3f}.pth"
    
    # best_model_path = f"{model_name}_model_epsilon_{best_epsilon:.3f}.pth"
    # torch.save(best_model.state_dict(), best_model_path)
    
    print(f"\nFinal Model => Loss: {best_loss:.3f}, Accuracy: {best_accuracy:.3f}, Epsilon: {best_epsilon:.3f}")
    return model #, epsilon


    # if is_canary: 
    #     best_model_path = f"./cifar_models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"

    #     # best_model_path = f"./cifar_models_100/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
    #     # best_model_path = f"./models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
    # else:
    #     best_model_path = f"./cifar_models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
    #     # best_model_path = f"./cifar_models_100/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
    #     # best_model_path = f"./models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
        
    # torch.save(best_model.state_dict(), best_model_path)
    # print(f"\nFinal Model => Loss: {best_loss:.3f}, Accuracy: {best_accuracy:.3f}, Epsilon: {best_epsilon:.3f}")
    # return model, epsilon

def train_model(model, dataloader, criterion, optimizer, num_epochs, model_name="model", is_dp=True, task_type='multiclass', target_epsilon=1,target_delta=1e-5, max_grad_norm=1, device='cpu', sample_num=1, is_canary=False):
    model.to(device)  
    best_accuracy = 0.0
    best_loss = 0
    best_model_path = None
    model.train()

    if is_dp:
        # delta = 1 / len(dataloader.dataset)
        privacy_engine = PrivacyEngine(accountant='rdp') #accountant=accountant)
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon, 
            target_delta=target_delta,  
            epochs=num_epochs
        )
        print(f"\nUsing epsilon={target_epsilon}, sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        epsilon = 0.0

        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            data = data.float()
            labels = labels.float().view(-1, 1) if task_type == 'binary' else labels.long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(output, labels, task_type=task_type)
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = running_accuracy / len(dataloader)
        print("epoch_accuracy: ", epoch_accuracy)
        if is_dp:
            epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}, Epsilon: {epsilon:.3f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_loss = epoch_loss
            best_model = model
            best_epsilon = epsilon if is_dp else 0
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # best_model_path = f"{model_name}_model_epsilon_{best_epsilon:.3f}.pth"
    if is_canary: 
        best_model_path = f"./cifar_models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"

        # best_model_path = f"./cifar_models_100/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
        # best_model_path = f"./models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
    else:
        best_model_path = f"./cifar_models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
        # best_model_path = f"./cifar_models_100/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
        # best_model_path = f"./models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
        
    torch.save(best_model.state_dict(), best_model_path)
    print(f"\nFinal Model => Loss: {best_loss:.3f}, Accuracy: {best_accuracy:.3f}, Epsilon: {best_epsilon:.3f}")
    return model, epsilon
    
def train_model(model, dataloader, criterion, optimizer, num_epochs, model_name="model", is_dp=True, task_type='multiclass', target_epsilon=1,target_delta=1e-5, max_grad_norm=1, device='cpu', sample_num=1, is_canary=False):
    model.to(device)  
    best_accuracy = 0.0
    best_loss = 0
    best_model_path = None
    model.train()

    if is_dp:
        # delta = 1 / len(dataloader.dataset)
        privacy_engine = PrivacyEngine(accountant='rdp') #accountant=accountant)
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon, 
            target_delta=target_delta,  
            epochs=num_epochs
        )
        print(f"\nUsing epsilon={target_epsilon}, sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        epsilon = 0.0

        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            data = data.float()
            labels = labels.float().view(-1, 1) if task_type == 'binary' else labels.long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(output, labels, task_type=task_type)
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = running_accuracy / len(dataloader)
        print("epoch_accuracy: ", epoch_accuracy)
        if is_dp:
            epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}, Epsilon: {epsilon:.3f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_loss = epoch_loss
            best_model = model
            best_epsilon = epsilon if is_dp else 0
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # best_model_path = f"{model_name}_model_epsilon_{best_epsilon:.3f}.pth"
    if is_canary: 
        best_model_path = f"./cifar_models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"

        # best_model_path = f"./cifar_models_100/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
        # best_model_path = f"./models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
    else:
        best_model_path = f"./cifar_models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
        # best_model_path = f"./cifar_models_100/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
        # best_model_path = f"./models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
        
    torch.save(best_model.state_dict(), best_model_path)
    print(f"\nFinal Model => Loss: {best_loss:.3f}, Accuracy: {best_accuracy:.3f}, Epsilon: {best_epsilon:.3f}")
    return model, epsilon


def train_model_with_memory_manager(model, dataloader, criterion, optimizer, num_epochs, model_name="model", is_dp=True, task_type='multiclass', target_epsilon=1, target_delta=1e-5, max_grad_norm=1, device='cpu', sample_num=1, is_canary=False, max_physical_batch_size=256):
    model.to(device)
    best_accuracy = 0.0
    best_loss = np.inf
    best_model_path = None
    model.train()

    if is_dp:
        # privacy_engine = PrivacyEngine(
        #     module=model,
        #     batch_size=max_physical_batch_size,
        #     sample_size=len(dataloader.dataset),
        #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        #     # noise_multiplier=optimizer.noise_multiplier,
        #     max_grad_norm=max_grad_norm,
        #     target_epsilon=target_epsilon,
        #     target_delta=target_delta,
        #     epochs=num_epochs
        # )
        # privacy_engine.attach(optimizer)

        privacy_engine = PrivacyEngine(accountant='rdp') #accountant=accountant)
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon, 
            target_delta=target_delta,  
            epochs=num_epochs
        )
        print(f"\nUsing epsilon={target_epsilon}, sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

    with BatchMemoryManager(data_loader=dataloader, max_physical_batch_size=max_physical_batch_size, optimizer=optimizer) as memory_safe_data_loader:
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_accuracy = 0.0

            for data, labels in memory_safe_data_loader:
                data, labels = data.to(device), labels.to(device)
                if task_type == 'binary':
                    labels = labels.float().view(-1, 1)
                else:
                    labels = labels.long()
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * data.size(0)
                running_accuracy += calculate_accuracy(output, labels, task_type=task_type) * data.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_accuracy = running_accuracy / len(dataloader.dataset)

            if is_dp:
                epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}, Epsilon: {epsilon:.3f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}")

            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_loss = epoch_loss
                best_model = model
                best_epsilon = epsilon if is_dp else 0
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # best_model_path = f"{model_name}_model_epsilon_{best_epsilon:.3f}.pth"
        if is_canary: 
            best_model_path = f"./cifar_models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"

            #best_model_path = f"./models_giant_epsilons/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
            # best_model_path = f"./models/canary/{model_name}_model_{sample_num}_canary_{timestamp}.pth"
        else:
            best_model_path = f"./cifar_models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
            # best_model_path = f"./models_giant_epsilons/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
            # best_model_path = f"./models/no_canary/{model_name}_model_{sample_num}_{timestamp}.pth"
            
    torch.save(best_model.state_dict(), best_model_path)
    print(f"\nFinal Model => Loss: {best_loss:.3f}, Accuracy: {best_accuracy:.3f}, Epsilon: {best_epsilon:.3f}")
    return model, epsilon




def calculate_accuracy(output, labels, task_type='multiclass'):
    if task_type == 'multiclass':
        # Convert output logits to predicted class indices
        predictions = torch.argmax(output, dim=1)
        # Ensure labels are in the correct shape
        labels = labels if labels.dim() == 1 else torch.argmax(labels, dim=1)
    else:
        # For binary classification (not directly applicable here, but for reference)
        predictions = output.round()

    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.size(0)  # Use size(0) to get the batch size
    return accuracy.item()

def evaluate_model(model, dataloader, criterion, device, task_type='multiclass'):
    model.eval()  
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            if task_type == 'multiclass':
                _, predicted = torch.max(outputs.data, 1)
            else:
                predicted = outputs.round()
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return accuracy, avg_loss