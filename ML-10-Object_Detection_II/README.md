# ML-10-Object Detection II

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Library](https://img.shields.io/badge/Library-Ultralytics%20YOLO-orange.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## Dataset

**Face Mask Detection** : https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

โปรเจกต์นี้ใช้ชุดข้อมูลภาพใบหน้าที่มีกรอบวัตถุ (Bounding Box) มาทำ Object Detection ด้วยโมเดล **Ultralytics YOLO** เพื่อแยกว่าบุคคลในภาพสวมหน้ากากอนามัยหรือไม่ ชุดข้อมูลมีภาพทั้งหมด **853 ภาพ** label เป็น Pascal VOC (XML) ซึ่งแปลงเป็น YOLO format เรียบร้อยแล้ว

### คลาสในชุดข้อมูล

| ID | ชื่อคลาส | ความหมาย |
|:---:|---|---|
| 0 | `with_mask` | สวมหน้ากากถูกต้อง |
| 1 | `without_mask` | ไม่สวมหน้ากาก |
| 2 | `mask_weared_incorrect` | สวมหน้ากากไม่ถูกต้อง |

---

## 📌 Project Overview

งานนี้เป็นการพัฒนาโมเดล Object Detection สำหรับตรวจจับหน้ากากอนามัยแบบเรียลไทม์ด้วย Ultralytics YOLO โดยโมเดลได้รับการฝึกฝน (Train) ให้สามารถจำแนกสถานะการสวมหน้ากากได้ครบถ้วนทั้ง 3 คลาส (`with_mask`, `without_mask`, `mask_weared_incorrect`) และสามารถนำมาทดสอบรันผ่านกล้องเว็บแคมได้อย่างแม่นยำ

---

## 📊 ผลการทดสอบและการแสดงผลผ่านเว็บแคม (Real-time Detection Results)

เมื่อทำการทดสอบผ่านสคริปต์ `realtest_test.py` โมเดลสามารถตรวจจับและแสดงกรอบพิกัดพร้อมค่าความมั่นใจ (Confidence Score) ได้อย่างแม่นยำในแต่ละสถานการณ์:

### 1. ไม่สวมหน้ากาก (`without_mask`)
![Without Mask](./สกรีนช็อต%202026-10-18%20003501.png)

### 2. สวมหน้ากากถูกต้อง (`with_mask`)
![With Mask](./สกรีนช็อต%202026-10-18%20004015.png)

### 3. สวมหน้ากากไม่ถูกต้อง / ไว้ใต้คาง (`mask_weared_incorrect`)
![Incorrect Mask](./สกรีนช็อต%202026-10-18%20004030.png)

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

📁 Project Structure
ML-10-Object Detection II/
├── dataset/                    # ข้อมูลสำหรับเทรน/ทดสอบ (YOLO format)
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── test/{images,labels}/
├── mask_detect/                # โฟลเดอร์เก็บน้ำหนักโมเดล (weights/best.pt)
├── realtime_test.py            # สคริปต์ทดสอบระบบแบบเรียลไทม์ผ่านเว็บแคม
├── image_test.py               # สคริปต์ทดสอบกับภาพนิ่ง
├── requirements.txt            # รายชื่อไลบรารีที่ต้องใช้
└── README.md                   # เอกสารประกอบโปรเจกต์