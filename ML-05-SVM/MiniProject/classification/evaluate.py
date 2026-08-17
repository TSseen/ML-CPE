import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, kernel_name, save_dir=None):

    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n------------ Evaluation ({kernel_name}) ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(
        y_test, predictions,
        labels=labels, target_names=classes, zero_division=0
    )
    print(report)

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion Matrix:")
    print(matrix)

    if save_dir:
        save_path = f"{save_dir}/confusion_matrix_{kernel_name}.png"
        plot_confusion_matrix(matrix, classes, kernel_name, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, kernel_name, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({kernel_name})")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_kernel_comparison(results, save_path):
    """results: {kernel_name: accuracy}"""

    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values(),
           color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Accuracy")
    ax.set_title("SVM Kernel Comparison")
    ax.set_ylim(0, 1)

    for i, (k, v) in enumerate(results.items()):
        ax.text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
