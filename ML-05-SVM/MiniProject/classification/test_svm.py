
import os
import json
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

IMG_SIZE = 64
N_SAMPLES = 4


def test_svm(n_samples=N_SAMPLES):

    # ใช้ kernel ที่ accuracy ดีที่สุดจาก main.py โดยอัตโนมัติ
    with open(f"{OUTPUT_DIR}/best_kernel.json") as f:
        best = json.load(f)
    best_kernel = best["best_kernel"]
    print(f"Using best kernel: {best_kernel}")

    # Load model and test set
    model = joblib.load(f"{OUTPUT_DIR}/svm_model_{best_kernel}.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # Pick random images (no seed -> different every run)
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # Predict
    predictions = model.predict(scaler.transform(X_sample))

    # Show results in a grid
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred}\nTrue: {true}", color=color, fontsize=9)

        print(f"[{i + 1}] Pred: {pred:<20} True: {true:<20} "
              f"{'OK' if correct else 'WRONG'}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction ({best_kernel} kernel): {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_svm()