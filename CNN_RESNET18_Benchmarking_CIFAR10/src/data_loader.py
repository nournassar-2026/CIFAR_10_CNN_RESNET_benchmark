# ============================================
# data_loader.py - Load CIFAR-10 from Drive
# ============================================

import sys
import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from config import DATA_PATH, BATCH_SIZE

def get_loaders():
    """Load CIFAR-10 from Google Drive"""
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading CIFAR-10 from Google Drive...")
    
    if os.path.exists(DATA_PATH):
        print("✅ Loading from Drive...")
        train_dataset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=False, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, download=False, transform=test_transform
        )
    else:
        print("❌ Data not found in Drive. Downloading...")
        train_dataset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, download=True, transform=test_transform
        )
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f" Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f" Batch size: {BATCH_SIZE}")
    
    return train_loader, val_loader, test_loader
