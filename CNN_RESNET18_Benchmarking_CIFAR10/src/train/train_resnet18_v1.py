# ============================================
# train_resnet18_v1.py - Train ResNet-18 V1 (Adam)
# ============================================

import sys
import os


# ✅ Auto-detect project root
def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    for _ in range(10):
        if os.path.exists(os.path.join(current_dir, 'config.py')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    common_paths = [
        '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10',
        './'
    ]
    for path in common_paths:
        if os.path.exists(os.path.join(path, 'config.py')):
            return path
    raise FileNotFoundError("config.py not found!")

project_root = get_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.optim as optim
import time
from config import device, BASE_PATH
from src.data_loader import get_loaders
from src.models import create_resnet18
from src.helpers import save_model, save_results, plot_curves, evaluate_model, train_loop


def train_resnet18_v1():
    """Train ResNet-18 V1 with Adam optimizer (no dropout, no weight decay)"""
    
    print("\n" + "="*60)
    print("TRAINING RESNET-18 V1 (Adam, lr=0.001, 20 epochs)")
    print("="*60)
    print("Architecture:")
    print("  - ResNet-18 (Pretrained on ImageNet)")
    print("  - Modified for CIFAR-10 (32x32 images)")
    print("  - No Dropout")
    print("  - No Weight Decay")
    print("  - Optimizer: Adam (lr=0.001)")
    print("="*60)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_loaders()
    
    # Create model with NO dropout (0.0)
    model = create_resnet18(dropout_rate=0.0).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model created with {total_params:,} parameters")
    
    # No weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 40
    patience = 5
    
    print(f"\n Training Configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Early Stopping Patience: {patience}")
    print(f"   - Optimizer: Adam (lr=0.001)")
    print(f"   - Weight Decay: 0 (None)")
    print(f"   - Dropout: 0.0 (None)")
    print(f"   - Loss: Cross Entropy")
    
    # Train with early stopping
    start_time = time.time()
    train_loss, val_loss, train_acc, val_acc, best_val_acc = train_loop(
        model, train_loader, val_loader, criterion, optimizer, epochs, device, patience
    )
    training_time = time.time() - start_time
    
    # Test
    test_acc = evaluate_model(model, test_loader, device)
    
    # Summary
    print(f"\n ResNet-18 V1 Test Accuracy: {test_acc:.2f}%")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Training Time: {training_time/60:.2f} min")
    print(f"   Epochs Trained: {len(train_loss)}/{epochs}")
    
    # Save model and results
    save_model(model, 'resnet18_v1.pth')
    
    results = {
        'model_type': 'ResNet-18 V1',
        'hyperparameters': {
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'weight_decay': 0,
            'dropout': 0.0,
            'epochs': epochs,
            'patience': patience,
            'batch_size': 64,
            'scheduler': 'None'
        },
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'training_time': training_time,
        'epochs_trained': len(train_loss),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    save_results(results, 'resnet18_v1_results')
    plot_curves(train_loss, val_loss, train_acc, val_acc, 'ResNet-18 V1 (Adam)', 'resnet18_v1_curves')
    # Auto-update comparison summary
    from helpers import update_comparison_summary
    update_comparison_summary()
    
    return test_acc, training_time


if __name__ == "__main__":
    train_resnet18_v1()