# ============================================
# evaluate_all.py - Test All Models
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

from src.test.test_cnn import test_cnn
from src.test.test_resnet18_v1 import test_resnet18_v1
from src.test.test_resnet18_v2 import test_resnet18_v2


def evaluate_all():
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)
    
    results = {}
    
    # Test all models
    results['CNN'] = test_cnn()
    results['ResNet-18 V1'] = test_resnet18_v1()
    results['ResNet-18 V2'] = test_resnet18_v2()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Test Accuracy':<20}")
    print("-"*40)
    for model, acc in results.items():
        if acc is not None:
            print(f"{model:<20} {acc:.2f}%")
        else:
            print(f"{model:<20} ❌ Not trained")
    
    return results

if __name__ == "__main__":
    evaluate_all()
