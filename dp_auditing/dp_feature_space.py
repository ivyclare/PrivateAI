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
from models import MNISTNet, wide_resnet16_4
from train import train_dp_model, calculate_accuracy, evaluate_model, train_model_with_memory_manager
from utils import create_canary_samples
from sklearn.decomposition import TruncatedSVD

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



def train_dp_models(model, dataloader, criterion, optimizer, num_epochs, model_name="model", is_dp=True, task_type='multiclass', target_epsilon=1,target_delta=1e-5, max_grad_norm=1, device='cpu'): #, sample_num=1, is_canary=False):
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

    sve_values, lsvr_values = [], []        
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
         
        if epoch % 10 == 0:  # This could be every epoch or every few epochs
            features = extract_features(model, dataloader, device)
            sve, lsvr = compute_svd_metrics(features)
            sve_values.append(sve)
            lsvr_values.append(lsvr)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # best_model_path = f"{model_name}_model_epsilon_{best_epsilon:.3f}.pth"
    
    # best_model_path = f"{model_name}_model_epsilon_{best_epsilon:.3f}.pth"
    # torch.save(best_model.state_dict(), best_model_path)
    
    print(f"\nFinal Model => Loss: {best_loss:.3f}, Accuracy: {best_accuracy:.3f}, Epsilon: {best_epsilon:.3f}")
    # return model #, epsilon
    return model, sve_values, lsvr_values

def extract_features(model, dataloader, device='cpu'):
    model.eval()
    features = []
    with torch.no_grad():
        for data, _ in dataloader:
            # Ensure data is on the same device as the model
            data = data.to(device)
            output = F.relu(model.bn1(model.conv1(data)))
            # Assuming you're extracting features from just after the first convolution and BN
            output = output.view(output.size(0), -1)  # Flatten the features if necessary
            features.append(output.cpu().numpy())  # Move the tensor back to CPU for numpy conversion
    features = np.concatenate(features, axis=0)
    return features


def compute_svd_metrics(features):
    svd = TruncatedSVD(n_components=min(features.shape)-1)
    svd.fit(features)
    singular_values = svd.singular_values_
    sve = -np.sum(np.log(singular_values / np.sum(singular_values)))  # Singular Value Entropy
    lsvr = np.log(np.max(singular_values) / np.sum(singular_values))  # Largest Singular Value Ratio
    return sve, lsvr

def adjust_state_dict(state_dict):
    """Adjusts the state_dict keys by removing the '_module.' prefix if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_module."):
            new_key = k[len("_module."):]  # Remove the prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

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
    # parser.add_argument('--num_canary_samples', type=float, default=10, help='Maximum gradient norm for clipping')
    # parser.add_argument('--file_number', type=int, required=True, help='File number for saving observations')


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("FILE NUM: ", args.file_number)
    print("Device: ", device)
    print(f"Epochs: {args.num_epochs} , Batch Size: {args.batch_size} , LR: {args.lr}")

    train_loader, test_loader, num_features, num_classes, input_shape = get_dataset(args.dataset_name, args.data_path, args.batch_size)
    
    non_dp_model_path = f"./models/{args.dataset_name}_model_non_dp.pth"
    dp_model_path = f"./models/{args.dataset_name}_model_dp.pth"

    criterion = nn.CrossEntropyLoss()
    
    # non-DP Model
    model = wide_resnet16_4().to(device)
    if os.path.exists(non_dp_model_path):
        print("\nLoading existing non-DP model...")
        model.load_state_dict(torch.load(non_dp_model_path))
        model.to(device)
    else:
        print("Training non-DP model...")
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        model, non_dp_sve_values, non_dp_lsvr_values = train_dp_models(model, train_loader, criterion, optimizer, args.num_epochs, model_name=args.dataset_name, is_dp=False, task_type='multiclass', target_epsilon=1, target_delta=1e-5, max_grad_norm=1, device=device)
        torch.save(model.state_dict(), non_dp_model_path)

    # print("\nPASSS\n")
    # DP Model
    dp_model = wide_resnet16_4().to(device) 
    dp_model = ModuleValidator.fix(dp_model)
    ModuleValidator.validate(dp_model, strict=False)
    # if os.path.exists(dp_model_path):
    #     print("Loading existing DP model...")
    #     dp_model.load_state_dict(torch.load(dp_model_path))
    #     dp_model.to(device)
    
    if os.path.exists(dp_model_path):
        print("Loading existing DP model...")
        state_dict = torch.load(dp_model_path)
        adjusted_state_dict = adjust_state_dict(state_dict)
        dp_model.load_state_dict(adjusted_state_dict)
        dp_model.to(device)
    else:
        print("Training DP model...")
        
        optimizer = optim.SGD(dp_model.parameters(), lr=args.lr)
        dp_model, dp_sve_values, dp_lsvr_values  = train_dp_models(dp_model, train_loader, criterion, optimizer, args.num_epochs, model_name=args.dataset_name, is_dp=True, task_type='multiclass', target_epsilon=args.target_epsilon, target_delta=args.target_delta, max_grad_norm=args.max_grad_norm, device=device)
        torch.save(dp_model.state_dict(), dp_model_path)
        # torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")
        # torch.save(dp_model. _module.state_dict(), dp_model_path)

    
    # model = train_dp_model(model, train_loader, criterion, optimizer, args.num_epochs, model_name=args.dataset_name, is_dp=False, task_type='multiclass', target_epsilon=1,target_delta=1e-5, max_grad_norm=1, device='cpu') #, sample_num=1, is_canary=False):
    # dp_model = train_dp_model(model, train_loader, criterion, optimizer, args.num_epochs, model_name=args.dataset_name, is_dp=True, task_type='multiclass', target_epsilon=1,target_delta=1e-5, max_grad_norm=1, device='cpu') #, sample_num=1, is_canary=False):

    # best_model_path = f"{args.dataset_name}_model.pth"
    print("\nExtracting Non-DP Features")
    features = extract_features(model, train_loader, device)
    sve, lsvr = compute_svd_metrics(features)
    print(f"Non-DP:\n\tSingular Value Entropy: {sve}, Largest Singular Value Ratio: {lsvr}")

    print("\nExtracting DP Features")
    dp_features = extract_features(dp_model, train_loader, device)
    dp_sve, dp_lsvr = compute_svd_metrics(dp_features)
    print(f"DP:\n\tSingular Value Entropy: {dp_sve}, Largest Singular Value Ratio: {dp_lsvr}")

    print(f"{non_dp_sve_values=}")
    print(f"{dp_sve_values=}")
    print(f"\n\n{non_dp_lsvr_values=}")
    print(f"{dp_lsvr_values=}")

    # Plotting
    epochs = list(range(1, args.num_epochs + 1))
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, non_dp_sve_values, label='Non-DP SVE')
    plt.plot(epochs, dp_sve_values, label='DP SVE')
    plt.title('Singular Value Entropy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('SVE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, non_dp_lsvr_values, label='Non-DP LSVR')
    plt.plot(epochs, dp_lsvr_values, label='DP LSVR')
    plt.title('Largest Singular Value Ratio Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('LSVR')
    plt.legend()

    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    main()
