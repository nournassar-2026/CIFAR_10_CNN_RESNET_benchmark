# ============================================
# compare.py - Compare All Models
# ============================================

import sys
import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import matplotlib.pyplot as plt
import numpy as np
from config import BASE_PATH

def load_results(filename):
    with open(f'{BASE_PATH}/results/{filename}.pkl', 'rb') as f:
        return pickle.load(f)

def compare_models():
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    try:
        cnn_results = load_results('cnn_results')
        resnet18_v1_results = load_results('resnet18_v1_results')
        resnet18_v2_results = load_results('resnet18_v2_results')
        print("✅ All results loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error loading results: {e}")
        print("   Please train all models first.")
        return
    
    print(f"\n{'Model':<20} {'Test Accuracy':<20} {'Training Time':<20} {'Epochs':<10}")
    print("-"*70)
    print(f"{'CNN':<20} {cnn_results['test_acc']:.2f}%{'':<17} {cnn_results['training_time']/60:.2f} min{'':<12} {cnn_results['epochs_trained']}")
    print(f"{'ResNet-18 V1':<20} {resnet18_v1_results['test_acc']:.2f}%{'':<17} {resnet18_v1_results['training_time']/60:.2f} min{'':<12} {resnet18_v1_results['epochs_trained']}")
    print(f"{'ResNet-18 V2':<20} {resnet18_v2_results['test_acc']:.2f}%{'':<17} {resnet18_v2_results['training_time']/60:.2f} min{'':<12} {resnet18_v2_results['epochs_trained']}")
    print("-"*70)
    
    models = ['CNN', 'ResNet-18 V1', 'ResNet-18 V2']
    accuracies = [cnn_results['test_acc'], resnet18_v1_results['test_acc'], resnet18_v2_results['test_acc']]
    best_idx = np.argmax(accuracies)
    print(f"\n🏆 Best Model: {models[best_idx]} with {accuracies[best_idx]:.2f}% accuracy")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars1 = ax1.bar(models, accuracies, color=colors)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Model Comparison: Test Accuracy')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12)
    
    times = [
        cnn_results['training_time']/60,
        resnet18_v1_results['training_time']/60,
        resnet18_v2_results['training_time']/60
    ]
    bars2 = ax2.bar(models, times, color=colors)
    ax2.set_ylabel('Training Time (minutes)')
    ax2.set_title('Model Comparison: Training Time')
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{t:.1f} min', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}/plots/model_comparison.png', dpi=150)
    plt.show()
    print(f"\n✅ Comparison plot saved: {BASE_PATH}/plots/model_comparison.png")
    
    with open(f'{BASE_PATH}/results/comparison_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Model':<20} {'Test Accuracy':<20} {'Training Time':<20}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'CNN':<20} {cnn_results['test_acc']:.2f}%{'':<17} {cnn_results['training_time']/60:.2f} min\n")
        f.write(f"{'ResNet-18 V1':<20} {resnet18_v1_results['test_acc']:.2f}%{'':<17} {resnet18_v1_results['training_time']/60:.2f} min\n")
        f.write(f"{'ResNet-18 V2':<20} {resnet18_v2_results['test_acc']:.2f}%{'':<17} {resnet18_v2_results['training_time']/60:.2f} min\n")
        f.write("-"*60 + "\n")
        f.write(f"\n🏆 Best Model: {models[best_idx]} with {accuracies[best_idx]:.2f}% accuracy\n")
    
    print(f"✅ Summary saved: {BASE_PATH}/results/comparison_summary.txt")

if __name__ == "__main__":
    compare_models()
