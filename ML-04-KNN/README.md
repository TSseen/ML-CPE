# ML-04 - K-Nearest Neighbors (KNN) & K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)


สร้างไปป์ไลน์ Machine Learning อย่างง่ายด้วยภาษา Python ครอบคลุมตั้งแต่การโหลดข้อมูล, การทำความสะอาดข้อมูล (Preprocessing), การทำ Feature Scaling, การฝึกสอนโมเดล (Model Training), การประเมินผล (Evaluation) และการทำนายผล

## Dataset
- **Wine Quality Dataset** (จาก Kaggle) https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
- จัดเก็บในโฟลเดอร์: `Dataset/WineQT.csv`

## 📂 Project Structure

```text
ML-LAB04-KNN/
├── Dataset/
│   └── WineQT.csv
├── classification/
│   ├── main.py
│   ├── data_loader.py
│   ├── knn_tf.py
│   ├── evaluate.py
│   └── outputs/
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       └── predictions.csv
├── clustering/
│   ├── main.py
│   ├── data_loader.py
│   ├── kmeans_tf.py
│   ├── knn_tools.py
│   ├── visualize.py
│   └── outputs/
│       ├── 01_elbow.png
│       ├── 02_clusters.png
│       ├── cluster_summary.csv
│       └── clustered_animals.csv
├── requirements.txt
└── link-data.txt
