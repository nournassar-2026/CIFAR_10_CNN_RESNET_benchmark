# ============================================
# train_cnn.py - Train CNN Model
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
from config import device, BASE_PATH, EPOCHS_CNN, PATIENCE
from src.data_loader import get_loaders
from src.models import CNN6
from src.helpers import save_model, save_results, plot_curves, evaluate_model, train_loop


def train_cnn():
    """Train CNN model with 6 conv layers"""
    
    print("\n" + "="*60)
    print("TRAINING CNN (6 Conv Layers)")
    print("="*60)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_loaders()
    
    # Create model
    model = CNN6().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model created with {total_params:,} parameters")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = EPOCHS_CNN
    patience = PATIENCE
    
    print(f"\n Training Configuration:")
    print(f"   - Max Epochs: {epochs}")
    print(f"   - Early Stopping Patience: {patience}")
    print(f"   - Optimizer: Adam (lr=0.001)")
    print(f"   - Loss: Cross Entropy")
    
    # Train
    start_time = time.time()
    train_loss, val_loss, train_acc, val_acc, best_val_acc = train_loop(
        model, train_loader, val_loader, criterion, optimizer, epochs, device, patience
    )
    training_time = time.time() - start_time
    
    # Test
    test_acc = evaluate_model(model, test_loader, device)
    
    # Summary
    print(f"\n CNN Test Accuracy: {test_acc:.2f}%")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Training Time: {training_time/60:.2f} min")
    print(f"   Epochs Trained: {len(train_loss)}/{epochs}")
    
    # Save model and results
    save_model(model, 'cnn_model.pth')
    
    results = {
        'model_type': 'CNN6',
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'training_time': training_time,
        'epochs_trained': len(train_loss),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    save_results(results, 'cnn_results')
    plot_curves(train_loss, val_loss, train_acc, val_acc, 'CNN', 'cnn_curves')
    # Auto-update comparison summary
    from helpers import update_comparison_summary
    update_comparison_summary()
    return model, test_acc


if __name__ == "__main__":
    train_cnn()
