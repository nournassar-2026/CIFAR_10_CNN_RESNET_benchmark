# ============================================
# main.py - Run Everything
# ============================================

import sys
import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.train.train_cnn import train_cnn
from src.train.train_resnet18_v1 import train_resnet18_v1
from src.train.train_resnet18_v2 import train_resnet18_v2
from src.test.evaluate_all import evaluate_all
from compare import compare_models

def main():
    print("="*60)
    print("CIFAR-10 MODEL BENCHMARK")
    print("CNN vs ResNet-18 V1 vs ResNet-18 V2")
    print("="*60)
    
    print("\n Running full pipeline...")
    print("   Training all models...")
    print("   Testing all models...")
    print("   Comparing results...")
    print("="*60)
    
    print("\n" + "="*60)
    print("STEP 1: TRAINING ALL MODELS")
    print("="*60)
    train_cnn()
    train_resnet18_v1()
    train_resnet18_v2()
    
    print("\n" + "="*60)
    print("STEP 2: TESTING ALL MODELS")
    print("="*60)
    evaluate_all()
    
    print("\n" + "="*60)
    print("STEP 3: COMPARING RESULTS")
    print("="*60)
    compare_models()
    
    print("\n" + "="*60)
    print(" FULL PIPELINE COMPLETE!")
    print("="*60)
    print("\n📁 Results saved to:")
    print("   - models/  (trained models)")
    print("   - results/ (training history)")
    print("   - plots/   (visualizations)")

if __name__ == "__main__":
    main()
