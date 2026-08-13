# ============================================
# test_cnn.py - Test CNN Model
# ============================================

import sys
import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import os

# Add parent directory to path
sys.path.append('/content/CNN_RESNET18_Benchmarking_CIFAR10')

import torch
from config import device, BASE_PATH
from src.data_loader import get_loaders
from src.models import CNN6
from src.helpers import evaluate_model, plot_confusion_matrix_with_stats

# CIFAR-10 class names
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def test_cnn():
    print("\n" + "="*60)
    print("TESTING CNN MODEL")
    print("="*60)
    
    # Load data
    _, _, test_loader = get_loaders()
    
    # Load model
    model = CNN6().to(device)
    model_path = f'{BASE_PATH}/models/cnn_model.pth'
    
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Model loaded: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found. Please train CNN first.")
        return None
    
    # Test accuracy
    test_acc = evaluate_model(model, test_loader, device)
    print(f"\n CNN Test Accuracy: {test_acc:.2f}%")
    
    # Confusion Matrix
    plot_confusion_matrix_with_stats(
        model, test_loader, CLASSES, device, 
        model_name='cnn', 
        save_dir=f'{BASE_PATH}/plots'
    )
    
    return test_acc

if __name__ == "__main__":
    test_cnn()
