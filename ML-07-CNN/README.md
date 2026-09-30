# ML-07-Convolutional Neural Network (CNN)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Library](https://img.shields.io/badge/Library-PyTorch%20%2F%20TensorFlow-orange.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## Dataset

- **UTKFace Dataset** (จาก Kaggle) [https://www.kaggle.com/datasets/jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)
  - ชุดข้อมูลภาพใบหน้าขนาดใหญ่ (กว่า 23,000 รูป) ที่มีความหลากหลายทั้งในเรื่องของช่วงอายุ (0-116 ปี), เพศ, และเชื้อชาติ 
  - เหมาะสำหรับนำมาใช้เทรนโมเดล CNN เพื่อทำ Image Classification, Age Estimation, หรือ Gender Detection

---

## 📌 Project Overview

โปรเจกต์นี้เป็นการพัฒนาระบบ Deep Learning สำหรับการทำ Image Classification และการทำนายคุณลักษณะจากภาพใบหน้า โดยใช้สถาปัตยกรรม **Convolutional Neural Network (CNN)** โค้ดถูกออกแบบให้ทำงานแบบ End-to-End Pipeline ซึ่งครอบคลุมตั้งแต่กระบวนการโหลดข้อมูลรูปภาพ (Data Loading), การจัดการภาพ (Preprocessing), การสร้างและเทรนโมเดล CNN, ไปจนถึงการประเมินประสิทธิภาพของโมเดลอย่างเป็นระบบ

---

## 📁 Project Structure

โครงสร้างของโฟลเดอร์ในโปรเจกต์นี้ถูกแบ่งการทำงานเป็นโมดูลย่อยๆ เพื่อให้ง่ายต่อการดูแลรักษา:

```text
ML-07-CNN/
└── mini-proj/
    ├── outputs/                 # โฟลเดอร์สำหรับเก็บผลลัพธ์ (เช่น กราฟ, Weights ของโมเดล, Logs)
    ├── cnn_model.py             # โครงสร้างสถาปัตยกรรมของ CNN Model (Layers ต่างๆ)
    ├── data_loader.py           # สคริปต์สำหรับจัดการการโหลด Dataset และสร้าง Data Batches
    ├── evaluate.py              # สคริปต์สำหรับประเมินประสิทธิภาพของโมเดล (เช่น ค่า Accuracy, Confusion Matrix)
    ├── main.py                  # ไฟล์สคริปต์หลักสำหรับรันกระบวนการ Training Pipeline
    ├── preprocessing.py         # สคริปต์สำหรับทำ Image Preprocessing และ Data Augmentation
    └── test_cnn.py              # สคริปต์สำหรับทดสอบการทำนายผล (Inference) รูปภาพใหม่
