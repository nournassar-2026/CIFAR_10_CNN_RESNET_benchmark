# ============================================
# config.py - Configuration Settings
# ============================================

import torch
import os

# Paths
BASE_PATH = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
DATA_PATH = '/content/drive/MyDrive/cifar10_data'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 64
EPOCHS_CNN = 30
EPOCHS_RESNET = 40
LEARNING_RATE_CNN = 0.001
LEARNING_RATE_RESNET = 0.001
PATIENCE = 5

# Create folders
os.makedirs(f'{BASE_PATH}/models', exist_ok=True)
os.makedirs(f'{BASE_PATH}/results', exist_ok=True)
os.makedirs(f'{BASE_PATH}/plots', exist_ok=True)

print(f" Using device: {device}")
print(f" Data path: {DATA_PATH}")
print(f" Base path: {BASE_PATH}")
