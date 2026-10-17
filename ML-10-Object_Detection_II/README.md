# ML-10-Object Detection II: Face Mask Detection with Ultralytics YOLO

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Library](https://img.shields.io/badge/Library-Ultralytics%20YOLO-orange.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## Dataset

**Face Mask Detection** : https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

โปรเจกต์นี้ใช้ชุดข้อมูลภาพใบหน้าที่มีกรอบวัตถุ (Bounding Box) มาทำ Object Detection เพื่อแยกว่าบุคคลในภาพสวมหน้ากากอนามัยหรือไม่ ชุดข้อมูลมีภาพทั้งหมด **853 ภาพ** ป้ายกำกับ (Label) เดิมเป็น Pascal VOC (XML) ซึ่งถูกแปลงเป็น YOLO format เรียบร้อยแล้ว

### คลาสในชุดข้อมูล

| ID | ชื่อคลาส | ความหมาย |
|:---:|---|---|
| 0 | `with_mask` | สวมหน้ากากอนามัยถูกต้อง |
| 1 | `without_mask` | ไม่สวมหน้ากากอนามัย |
| 2 | `mask_weared_incorrect` | สวมหน้ากากไม่ถูกต้อง (เช่น สวมไว้ใต้คาง) |

---

## 📌 Project Overview

งานนี้เป็นการพัฒนาต่อยอดจากโปรเจกต์ตรวจจับวัตถุเบื้องต้น โดยยกระดับมาใช้สถาปัตยกรรมสมัยใหม่ **Ultralytics YOLO** สำหรับตรวจสอบการสวมหน้ากากอนามัยแบบเรียลไทม์ โมเดลได้รับการฝึกฝน (Train) ให้มีความสามารถในการจำแนกสถานะทั้ง 3 คลาสได้อย่างแม่นยำ แก้ไขข้อจำกัดจากโมเดลรุ่นก่อนหน้า (เช่น ปัญหาการตีความกรอบวัตถุเกินจริงบริเวณพื้นหลังหรือผนัง) ทำให้ระบบสามารถตรวจจับใบหน้าและแสดงผลพร้อมค่าความมั่นใจ (Confidence Score) ได้อย่างมีประสิทธิภาพผ่านกล้องเว็บแคม

---


## 📊 ผลการทดสอบและการแสดงผลผ่านเว็บแคม (Real-time Detection Results)

เมื่อทำการทดสอบผ่านสคริปต์ `realtime_test.py` โมเดลสามารถตรวจจับและแสดงกรอบพิกัดพร้อมค่าความมั่นใจ (Confidence Score) ได้อย่างแม่นยำในแต่ละสถานการณ์:

### 1. ไม่สวมหน้ากาก (`without_mask`)
[![Without Mask](./without_mask.png)](https://github.com/TSseen/ML-CPE/blob/main/ML-10-Object_Detection_II/mini-proj/without_mask.png)
> *ผลการทดสอบการตรวจจับกรณีไม่สวมหน้ากากอนามัย (ความมั่นใจ 0.88)*

### 2. สวมหน้ากากถูกต้อง (`with_mask`)
[![With Mask](./with_mask.png)](https://github.com/TSseen/ML-CPE/blob/main/ML-10-Object_Detection_II/mini-proj/with_mask.png)
> *ผลการทดสอบการตรวจจับกรณีสวมหน้ากากอนามัยอย่างถูกต้อง (ความมั่นใจ 0.87)*

### 3. สวมหน้ากากไม่ถูกต้อง / ไว้ใต้คาง (`mask_weared_incorrect`)
[![Incorrect Mask](./incorrect_mask.png)](https://github.com/TSseen/ML-CPE/blob/main/ML-10-Object_Detection_II/mini-proj/incorrect_mask.png)
> *ผลการทดสอบการตรวจจับกรณีสวมหน้ากากไว้ใต้คาง (ความมั่นใจ 0.87)*

---

## 🚀 How to Use

```bash
# 1. เตรียม environment และเปิดใช้งาน Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. ทดสอบโมเดลกับภาพนิ่ง
python image_test.py

# 3. ทดสอบแบบเรียลไทม์ผ่านเว็บแคม (กด 'q' เพื่อออก)
python realtime_test.py
```
## 📁 Project Structure

```
mini-proj/
├── dataset/                    # ข้อมูลสำหรับเทรนและทดสอบ (YOLO format)
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── test/{images,labels}/
├── mask_detect/                # โฟลเดอร์เก็บไฟล์น้ำหนักโมเดล (weights/best.pt)
├── evaluate.py                 # สคริปต์ประเมินประสิทธิภาพโมเดล
├── image_test.py               # สคริปต์ทดสอบโมเดลกับภาพนิ่ง
├── realtime_test.py            # สคริปต์ทดสอบโมเดลแบบเรียลไทม์ผ่านเว็บแคม
├── train.py                    # สคริปต์สำหรับการเทรนโมเดล YOLO
├── requirements.txt            # รายชื่อไลบรารีที่จำเป็นต้องใช้งาน
├── without_mask.png            # ภาพผลลัพธ์การทดสอบคลาส without_mask
├── with_mask.png               # ภาพผลลัพธ์การทดสอบคลาส with_mask
├── incorrect_mask.png          # ภาพผลลัพธ์การทดสอบคลาส mask_weared_incorrect
└── README.md                   # เอกสารอธิบายโปรเจกต์
```
