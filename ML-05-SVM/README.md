# ML-05-Support Vector Machine (SVM) |

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)



</div>

---

## 📌 Project Overview
This project implements an end-to-end Machine Learning pipeline following standard data science workflows: loading image data, preprocessing, feature scaling, model training with multiple SVM kernels, and comprehensive evaluation.

---

## 🗂️ Project Structure

```text
ML-05-SVM/
└── MiniProject/
    └── classification/
        ├── outputs/              # Generated models, plots, and metrics
        │   ├── best_kernel.json
        │   ├── classes.json
        │   ├── confusion_matrix_linear.png
        │   ├── confusion_matrix_poly.png
        │   ├── confusion_matrix_rbf.png
        │   ├── kernel_comparison.png
        │   └── prediction_sample.png
        ├── data_loader.py        # Loads and reads images from directories
        ├── preprocessing.py      # Resizes and flattens image arrays
        ├── split_data.py         # Splits data into train/test sets
        ├── svm_model.py          # Trains SVM models (Linear, Poly, RBF)
        ├── evaluate.py           # Generates metrics and confusion matrices
        ├── main.py               # Main execution script
        └── test_svm.py           # Unit testing for pipeline components
