# ============================================
# helpers.py - Helper Functions
# ============================================

import sys
import os

# Add project root to path
project_root = '/content/drive/MyDrive/CNN_RESNET18_Benchmarking_CIFAR10'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from config import BASE_PATH, device

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def save_model(model, filename):
    torch.save(model.state_dict(), f'{BASE_PATH}/models/{filename}')
    print(f" Model saved: {BASE_PATH}/models/{filename}")

def save_results(results, filename):
    with open(f'{BASE_PATH}/results/{filename}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f" Results saved: {BASE_PATH}/results/{filename}.pkl")

def load_results(filename):
    with open(f'{BASE_PATH}/results/{filename}.pkl', 'rb') as f:
        return pickle.load(f)

def plot_curves(train_loss, val_loss, train_acc, val_acc, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_acc, label='Training Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}/plots/{filename}.png', dpi=150)
    plt.show()
    print(f"✅ Plot saved: {BASE_PATH}/plots/{filename}.png")

def plot_confusion_matrix(model, test_loader, classes, device, save_path=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model.__class__.__name__}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Confusion matrix saved: {save_path}")
    
    return fig, cm

def print_classification_report(model, test_loader, classes, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

def plot_confusion_matrix_with_stats(model, test_loader, classes, device, model_name, save_dir=None):
    print(f"\n Analyzing {model_name}...")
    
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/{model_name}_confusion_matrix.png'
    
    fig, cm = plot_confusion_matrix(model, test_loader, classes, device, save_path)
    plt.show()
    
    report = print_classification_report(model, test_loader, classes, device)
    
    return fig, cm, report

def train_loop(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=5, scheduler=None):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
                  f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%, '
                  f'LR: {current_lr:.6f}')
        else:
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
                  f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')
        
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
        else:
            patience_counter += 1
            
        
        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    return train_losses, val_losses, train_accuracies, val_accuracies, best_val_acc

def update_comparison_summary():
    """Auto-update comparison summary after training"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from compare import compare_models
        print("\n Updating comparison summary...")
        compare_models()
        print(" Comparison summary updated!")
    except Exception as e:
        print(f" Could not update comparison: {e}")