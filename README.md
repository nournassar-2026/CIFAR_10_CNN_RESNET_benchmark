# CIFAR_10_CNN_RESNET_benchmark
Benchmarking CNN vs Fine-tuned ResNet on CIFAR-10, the CNN model achieved 84.12% test accuracy, while ResNet18_v1  test accuracy 90.86% . The best model ResNet18_v2 achieved 94.88%  .
# CNN vs ResNet-18 Benchmarking on CIFAR-10


 ## 📊 Dataset
This project uses the CIFAR-10 dataset containing:

  * 60,000 32×32 color images

  * 10 classes: 
    airplane, automobile, bird, cat, deer

   dog, frog, horse, ship, truck

 * The dataset is automatically downloaded to the data/ folder or loaded from Google Drive.
 * The dataset can be downloaded from https://cave.cs.toronto.edu/kriz/cifar.html
 * The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
 
  * Examples:
 ![ ](image.png)  ![alt text](image-1.png) ![alt text](image-2.png) ![alt text](image-3.png) ![alt text](image-4.png)
  
## 📋 Project Overview

 #### This project benchmarks three different models on the CIFAR-10 dataset:
- **CNN** (6 Convolutional Layers, 0.5 Dropout)
- **ResNet-18 V1** (Pretrained, Adam Optimizer)
- **ResNet-18 V2** (Pretrained, SGD Optimizer, CosineAnnealingLR)

#### The project includes complete training, testing, evaluation scripts.

---

## 📊 Results Summary

| Model            | Test Accuracy | Training Time | Parameters |
|------------------|---------------|---------------|------------|
| **CNN**          | 84.12%          | 21 min       | ~1.2M      |
| **ResNet-18 V1** | 90.86%          | 23 min       | ~11.2M     |
| **ResNet-18 V2** |94.88%          | 43 min       | ~11.2M     |

### 🏆Best Model: ResNet-18 V2 with 94.58% accuracy 

---

## 📁 Project Structure
```
CNN_RESNET18_Benchmarking
│
├── README.md # This file
├── requirements.txt # Dependencies
│
├── config.py # Configuration settings
├── compare.py # Compare all models
├── main.py # Run everything
│
├── src/
│ ├── data_loader.py # CIFAR-10 loading
│ ├── helpers.py # Helper functions
│ ├── models.py # Model architectures
│ │
│ ├── train/
│ │ ├─ train_cnn.py # Train CNN
│ │ ├─train_resnet18_v1.py #Train ResNet-18 V1
│ │ └─train_resnet18_v2.py #Train ResNet-18 V2
│ │
│ └── test/
│ ├── test_cnn.py # Test CNN
│ ├── test_resnet18_v1.py # Test ResNet-18 V1
│ ├── test_resnet18_v2.py # Test ResNet-18 V2
│ └── evaluate_all.py # Test all models
│
├── models/ # Trained models (.pth)
├── results/ # Results (.pkl)
└── plots/ # Visualizations (.png)

```
---
## Implementation:
  #### The project was implemented using Google Colab , and the free GBU has been used.
  
## 🚀 Quick Start
#### You can download the main folder project to your drive, then you can run using Colab, then mount drive to colab
```
from google.colab import drive
drive.mount('/content/drive')
```
```
import os

# Set project path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
```

### Option 1: Run Everything

```bash
!python main.py
```
### Option 2: Train Specific Model
```bash
# Train CNN only
!python main.py src/train/train_cnn.py

# Train ResNet-18 V1 only
!python src/train/train_resnet18_v1.py

# Train ResNet-18 V2 only
!python src/train/train_resnet18_v2.py
```
### Option 3: Test All Models
```bash
!python src/test/evaluate_all.py
```
#### Then You need to run compare.py to update results.
```bash
!python compare.py
```

 
  
## 🏗️ Model Architectures
### CNN (6 Layers)
``` 
Input (32×32×3)
    ↓
Conv1 (3→32, 3×3) + BN + ReLU + MaxPool
    ↓
Conv2 (32→64, 3×3) + BN + ReLU + MaxPool
    ↓
Conv3 (64→128, 3×3) + BN + ReLU + MaxPool
    ↓
Conv4 (128→256, 3×3) + BN + ReLU
    ↓
Conv5 (256→512, 3×3) + BN + ReLU
    ↓
Conv6 (512→512, 3×3) + BN + ReLU
    ↓
Flatten (8192)
    ↓
Dropout (0.5) + FC1 (512) + ReLU
    ↓
Dropout (0.5) + FC2 (10)
```
### ResNet-18 (Pretrained on ImageNet)
 * Modified conv1: kernel_size=3, stride=1,  padding=1

 * Removed maxpool (for 32×32 images)

 * FC layer: 10 output classes
  
## 📈 Training Configuration


|Parameter |CNN| ResNet-18 V1 |	ResNet-18 V2|
|------------------|--------------|---------------|---------------|
|Epochs	|30|	40|	40|
|Batch Size	|64	|64|	64|
|Optimizer|	Adam	|Adam	|SGD||
Learning Rate	|0.001	|0.001	|0.01
|Weight Decay  	|0	|0	|0.0002
|Dropout	|0.5	|0.0|	0.3
|Scheduler|	None	|None	|CosineAnnealingLR
|Early Stopping|	Yes (patience=5)|	Yes (patience=5)	|Yes (patience=5)

 ### 🧪Testing & Evaluation
#### Each test script generates:

 * Test accuracy

 * Confusion matrix (saved as PNG)
  
 * Classification report (precision, recall, f1-score)
  
#### Run All Tests
```
!python src/test/evaluate_all.py
```
#### Test Individual Model
```
# Test CNN
!python src/test/test_cnn.py

# Test ResNet-18 V1
!python src/test/test_resnet18_v1.py

# Test ResNet-18 V2
!python src/test/test_resnet18_v2.py
```



## 📁 Output Files
### Models (models/)
```
cnn_model.pth  # CNN weights

resnet18_v1.pth # ResNet-18 V1 weights

resnet18_v2.pth # ResNet-18 V2 weights
```
### Results (results/)
```
cnn_results.pkl  # CNN training history

resnet18_v1_results.pkl # ResNet-18 V1 history

resnet18_v2_results.pkl # ResNet-18 V2 history

comparison_summary.txt # Model comparison
```
### Plots (plots/)

![CNN training curves](plots/cnn_curves.png)


![ResNet-18 V1 Curves](https://raw.githubusercontent.com/nournassar-2026/CIFAR_10_CNN_RESNET_benchmark/main/plots/resnet18_v1_curves.png)

![alt text](https://raw.githubusercontent.com/nournassar-2026/CIFAR_10_CNN_RESNET_benchmark/main/plots/    resnet18_v2_curves.png)

![Model Comparison](https://raw.githubusercontent.com/nournassar-2026/CIFAR_10_CNN_RESNET_benchmark/main/plots/model_comparison.png)
```
*_confusion_matrix.png # Confusion matrices
```
![ ](cnn_confusion_matrix-2.png) 

![alt text](resnet18_v1_confusion_matrix.png)
![alt text](resnet18_v2_confusion_matrix.png)

##  🐛Debugging 
## Problems encountered during this project and how they were fixed:
  ### 📋 Problem 1: ModuleNotFoundError - No module named 'config'
  
  ```
  ModuleNotFoundError: No module named 'config'
```
#### When trying to run training scripts, Python couldn't find config.py
#### Fix: 
#### Added this code at the top of every script:
```
import sys
import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)
```
### 📋 Problem 2: Results Not Updating After Retraining

#### After retraining ResNet-18 V1, the comparison summary still showed old accuracy (88.11% instead of 90.81%).

#### Fix:
#### Added this to each training script after saving results
```
def update_comparison_summary():
    """Auto-update comparison summary after training"""
    try:
        from compare import compare_models
        print("\n🔄 Updating comparison summary...")
        compare_models()
        print("✅ Comparison summary updated!")
    except Exception as e:
        print(f"⚠️ Could not update comparison: {e}")

# Call after saving results
update_comparison_summary()
```
### 📋 Problem 3: Confusion Matrix Not Saved

#### Confusion matrix plot was showing but not saving to file, and no error message was displayed.
#### Fix
```
# Before (didn't save)
plot_confusion_matrix_with_stats(
    model, test_loader, CLASSES, device, 
    model_name='cnn'
)

# After (saves to file)
plot_confusion_matrix_with_stats(
    model, test_loader, CLASSES, device, 
    model_name='cnn', 
    save_dir=f'{BASE_PATH}/plots'  # ✅ Added save directory
)
```
### 📋 Problem 4: ResNet Models Not Converging
#### ResNet-18 V2 was stuck at 83% accuracy and not improving after 20 epochs.
#### Root Cause:
* Learning rate (0.1) was too high for SGD

* Cosine annealing scheduler was reducing LR too quickly

* Weight decay was too strong

#### Fix
```
# Before (poor performance)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

# After (improved performance)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0002)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
```

## 📝 License
### This project is open-source and available under the MIT License.
