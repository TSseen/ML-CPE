import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_model_comparison(histories_dict, output_path='outputs/model_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    markers = ['o', 's', '^', 'D', 'v', '*', 'x']
    
    # --- กราฟซ้าย: (a) Training accuracy of models. ---
    for i, (name, history) in enumerate(histories_dict.items()):
        final_acc = history.history['accuracy'][-1] * 100
        label_text = f"{name}: {final_acc:.2f}%"
        axes[0].plot(history.history['accuracy'], label=label_text, marker=markers[i % len(markers)], markersize=5)
        
    axes[0].set_title('(a) Training accuracy of models.')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # --- กราฟขวา: (b) Validation accuracy of models. ---
    for i, (name, history) in enumerate(histories_dict.items()):
        final_val_acc = history.history['val_accuracy'][-1] * 100
        label_text = f"{name}: {final_val_acc:.2f}%"
        axes[1].plot(history.history['val_accuracy'], label=label_text, marker=markers[i % len(markers)], markersize=5)
        
    axes[1].set_title('(b) Validation accuracy of models.')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"บันทึกกราฟเปรียบเทียบแบบอาจารย์ไว้ที่ {output_path}")

def plot_and_save_confusion_matrix(model, val_ds, class_names, model_name=""):
    y_true = []
    y_pred = []
    
    print(f"กำลังสร้าง Confusion Matrix สำหรับโมเดล {model_name}...")
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {model_name}')
    
    output_path = f'outputs/confusion_matrix_{model_name}.png' if model_name else 'outputs/confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"บันทึก Confusion Matrix ของ {model_name} ไว้ที่ {output_path}")