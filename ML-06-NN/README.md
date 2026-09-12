# ML-06-Neural Network (NN)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
</div>

Build a simple NN pipeline using Python for image recognition. The project covers image loading, preprocessing, dataset splitting, neural network training, evaluation, and prediction.

# Data 
# 🎭 Facial Expression Recognition with Neural Networks (FER-2013)
FER-2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013?select=test
About Dataset
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.


# Structure

```text
ML-06-NN/
├── data/                    # ชุดข้อมูลที่ใช้ในการทดลอง
├── notebooks/
│   └── ML_06_NN.ipynb      # โน้ตบุ๊กแสดงขั้นตอนการทดลองและโค้ดทั้งหมด
├── models/                  # Checkpoints หรือโมเดลที่เทรนเสร็จสมบูรณ์ (.pt/.h5)
├── requirements.txt         # ไลบรารีและ Dependencies
└── README.md                # รายละเอียดและสรุปผลการทดลอง
```
## 📌 บทคัดย่อ (Summary)

โครงงานนี้พัฒนาแบบจำลองโครงข่ายประสาทเทียม (Neural Network) เพื่อจำแนกอารมณ์จากภาพใบหน้ามนุษย์โดยใช้ชุดข้อมูล **FER-2013 (Facial Expression Recognition 2013)** ซึ่งประกอบด้วย 7 อารมณ์หลัก ได้แก่:
`Angry` (โกรธ), `Disgust` (รังเกียจ), `Fear` (กลัว), `Happy` (มีความสุข), `Sad` (เศร้า), `Surprise` (ประหลาดใจ) และ `Neutral` (ปกติ)

### ลำดับขั้นตอนการทำงาน:
* **Automated Data Loading & Preprocessing:** โหลดไฟล์ภาพใบหน้าจากโครงสร้างไดเรกทอรีแต่ละคลาสโดยอัตโนมัติ ทำการปรับขนาดภาพ (Resizing) ให้มีขนาดคงที่ และแปลงระบบสีจาก **BGR เป็น RGB** เพื่อความถูกต้องของช่องสัญญาณสี
* **Dataset Partitioning:** แบ่งชุดข้อมูลออกเป็น **Training Set, Validation Set และ Test Set** เพื่อใช้ในการฝึกสอนและทดสอบโมเดลอย่างเป็นกลาง
* **Comprehensive Evaluation:** ประเมินประสิทธิภาพโมเดลอย่างละเอียดด้วยค่า **Accuracy, Precision, Recall, F1-Score**, แสดง **Confusion Matrix** เพื่อวิเคราะห์ความแม่นยำรายอารมณ์ และพล็อตกราฟ **Training History** (Loss & Accuracy)

## 🔄 ลำดับขั้นตอนการประมวลผล (Workflow Pipeline)
```text
[FER-2013 Dataset (7 Emotion Classes)]
               │
               ▼ (OpenCV Preprocessing)
[Auto Load ➔ Resize Resolution ➔ BGR to RGB ➔ Normalization]
               │
               ▼ (Dataset Splitting)
┌──────────────────────┬──────────────────────┬──────────────────────┐
│     Training Set     │    Validation Set    │       Test Set       │
└──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
           │                      │                      │
           ▼                      ▼                      │
┌─────────────────────────────────────────┐              │
│    Neural Network Model Training        │              │
└────────────────────┬────────────────────┘              │
                     │                                   │
                     ▼ (Inference & Evaluation)          ▼
            [Multi-class Evaluation] ◄───────────────────┘
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[Classification Report & Matrix] [Loss / Accuracy History Plots]

```



