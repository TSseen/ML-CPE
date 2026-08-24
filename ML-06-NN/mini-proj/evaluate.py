import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(y_test, predictions, classes, save_path=None):
    """ประเมินผลและสร้าง Confusion Matrix"""
    print("\nกำลังสร้าง Confusion Matrix...")
    cm = confusion_matrix(y_test, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"บันทึก Confusion Matrix ไว้ที่: {save_path}")
        
    plt.close(fig)

def plot_history(history, save_path=None):
    """วาดกราฟแสดงค่า Loss และ Accuracy ระหว่างการเทรน"""
    print("กำลังสร้างกราฟ Training History...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # กราฟ Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # กราฟ Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"บันทึกกราฟ Training History ไว้ที่: {save_path}")
        
    plt.close(fig)