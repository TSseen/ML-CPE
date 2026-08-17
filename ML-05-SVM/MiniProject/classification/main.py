import json
import os

import joblib
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from svm_model import train_svm_all_kernels, predict_svm
from evaluate import evaluate_model, plot_kernel_comparison

# ปรับ path นี้ให้ตรงกับตำแหน่งที่แตกไฟล์จริง
# บน Kaggle Notebook: "/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset"
DATA_PATH = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-05-SVM\simpsons_dataset"

# OUTPUT_DIR หาตำแหน่งตัวเองอัตโนมัติ (โฟลเดอร์ "outputs" ข้างๆ ไฟล์นี้เสมอ)
# ไม่ว่าจะรันจากที่ไหนก็ตาม เพื่อให้ตรงกับ test_svm.py เสมอ
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
IMG_SIZE = 64
TEST_SIZE = 0.2
MAX_PER_CLASS = 300


def main():

    print("--" * 30)
    print("SVM Image Classification: Simpsons Characters")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print(f"\nTotal images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing -> feature vectors
    print("\n[Step 2] Converting images to feature vectors...")
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Step 4: Train SVM with all 3 kernels (also standardizes features)
    print("\n[Step 4] Training SVM (linear, poly, rbf)...")
    models, scaler = train_svm_all_kernels(X_train, y_train)
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    # Step 5 + 6: Predict and evaluate each kernel
    print("\n[Step 5-6] Testing and evaluating each kernel...")
    results = {}
    for kernel, model in models.items():
        predictions = predict_svm(model, scaler, X_test)
        acc = evaluate_model(y_test, predictions, classes, kernel,
                              save_dir=OUTPUT_DIR)
        results[kernel] = acc
        joblib.dump(model, f"{OUTPUT_DIR}/svm_model_{kernel}.pkl")

    # Step 7: Compare kernels
    print("\n[Step 7] Comparing kernels...")
    plot_kernel_comparison(results, f"{OUTPUT_DIR}/kernel_comparison.png")
    for k, v in results.items():
        print(f"{k:10s}: {v*100:.2f}%")

    best_kernel = max(results, key=results.get)
    print(f"\nBest kernel: {best_kernel} ({results[best_kernel]*100:.2f}%)")

    with open(f"{OUTPUT_DIR}/best_kernel.json", "w") as f:
        json.dump({"best_kernel": best_kernel, "accuracy": results[best_kernel]}, f)


if __name__ == "__main__":
    main()