# ============================================
# test_resnet18_v1.py - Test ResNet-18 V1
# ============================================

import sys
import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
from config import device, BASE_PATH
from src.data_loader import get_loaders
from src.models import create_resnet18
from src.helpers import evaluate_model, plot_confusion_matrix_with_stats

# CIFAR-10 class names
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def test_resnet18_v1():
    print("\n" + "="*60)
    print("TESTING RESNET-18 V1 MODEL")
    print("="*60)
    
    _, _, test_loader = get_loaders()
    
    # ✅ Match the saved model: NO dropout (0.0)
    model = create_resnet18(dropout_rate=0.0).to(device)
    model_path = f'{BASE_PATH}/models/resnet18_v1.pth'
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found. Please train ResNet-18 V1 first.")
        return None
    except RuntimeError as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tip: Make sure the model architecture matches the saved model.")
        print("   The saved model likely has NO dropout in the FC layer.")
        return None
    
    test_acc = evaluate_model(model, test_loader, device)
    print(f"\n📊 ResNet-18 V1 Test Accuracy: {test_acc:.2f}%")
    
    plot_confusion_matrix_with_stats(
        model, test_loader, CLASSES, device, 
        model_name='resnet18_v1', 
        save_dir=f'{BASE_PATH}/plots'
    )
    
    return test_acc

if __name__ == "__main__":
    test_resnet18_v1()
