# ML-09-Object Detection I

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Library](https://img.shields.io/badge/Library-TensorFlow%20%2F%20Keras-orange.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## Dataset

**Face Mask Detection** : https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

โปรเจกต์นี้ใช้ชุดข้อมูลภาพใบหน้าที่มีกรอบวัตถุ (Bounding Box) มาทำ Object Detection เพื่อแยกว่าบุคคลในภาพสวมหน้ากากอนามัยหรือไม่ แบ่งเป็น 3 คลาส ได้แก่ with_mask (สวมหน้ากากถูกต้อง), without_mask (ไม่สวมหน้ากาก) และ mask_weared_incorrect (สวมหน้ากากไม่ถูกต้อง) ชุดข้อมูลมีภาพทั้งหมด 853 ภาพ label เป็น Pascal VOC (XML) ซึ่งสคริปต์ `convert.py` จะแปลงเป็น YOLO format และแบ่งเป็นชุด Train / Validation / Test ในสัดส่วน 70 / 15 / 15 (สุ่มด้วย seed 42 เพื่อให้ได้ผลซ้ำเดิม) ในชุดทดสอบมี crop ทั้งหมด 551 รูป แบ่งเป็น 461, 74 และ 16 รูปตามลำดับ จำนวนแต่ละคลาสไม่บาลานซ์กันอย่างมาก เราจึงดูค่า Macro F1 ควบคู่กับ Accuracy ไม่ใช้ Accuracy อย่างเดียว

---

## 📌 Project Overview

งานนี้เป็นการทดลองสถาปัตยกรรม R-CNN และ Fast R-CNN ด้วย Python และ TensorFlow/Keras โดยใช้โค้ดต้นแบบจากอาจารย์ซึ่งเดิมใช้ชุดข้อมูล BDD100K แล้วเปลี่ยนมาใช้ชุดข้อมูลหน้ากากแทน ทั้งสองโมเดลเป็นเวอร์ชันง่ายเพื่อการเรียนรู้ (crop-classification baseline) คือรับภาพที่ตัดจากกรอบมาจำแนกว่าเป็นคลาสไหน ไม่มี Region Proposal, Bounding-box Regression และ NMS แบบฉบับสมบูรณ์ เราเทรนทั้งสองโมเดลด้วยข้อมูล Train เต็มชุดและจำนวน Epoch เท่ากันที่ 10 แล้ววัดผลจาก Accuracy, Precision, Recall และ F1 บนชุดทดสอบ

---

## 📊 ผลการทดลองและการวิเคราะห์

### เปรียบเทียบภาพรวม

| โมเดล | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| R-CNN | 90.38% | 0.55 | 0.59 | 0.57 |
| Fast R-CNN | 92.74% | 0.73 | 0.60 | 0.63 |

### R-CNN

R-CNN ได้ Accuracy 90.38% และ Macro F1 0.57 คลาส with_mask ทำได้ดีมาก (Precision 0.94, Recall 0.95, F1 0.94) เพราะเป็นคลาสที่มีข้อมูลมากที่สุด คลาส without_mask จับได้ค่อนข้างครบ (Recall 0.84) แต่ทายผิดบ่อยกว่า (Precision 0.70, F1 0.76) ส่วนคลาส mask_weared_incorrect ซึ่งมีเพียง 16 ตัวอย่างในชุดทดสอบ โมเดลทายถูกไม่ได้เลย ทั้ง Precision, Recall และ F1 เป็น 0.00 จุดนี้เองที่ทำให้ Macro F1 ต่ำกว่า Accuracy มาก

### Fast R-CNN

Fast R-CNN ได้ Accuracy 92.74% และ Macro F1 0.63 สูงกว่า R-CNN เล็กน้อย คลาส with_mask ดีขึ้นเป็น F1 0.96 (Precision 0.93, Recall 0.99) คลาส without_mask ทายแม่นขึ้นชัดเจน (Precision 0.92, F1 0.82) แต่จับได้ไม่ครบเท่าเดิม (Recall 0.74 เทียบกับ 0.84 ของ R-CNN) ส่วนคลาส mask_weared_incorrect ทำได้เพียงเล็กน้อย (Precision 0.33, Recall 0.06, F1 0.11) ซึ่งยังต่ำมาก

### ข้อสังเกตจากการทดสอบด้วยเว็บแคม

เมื่อทดสอบด้วย `realtime_test.py` ทั้งสองโมเดลแสดงกรอบเกินจำนวนมากเต็มหน้าจอ แม้บริเวณที่ไม่ใช่ใบหน้า เช่น ผนัง เตียง หรือเสื้อ ก็ถูกทายเป็นหนึ่งในสามคลาสด้วยความมั่นใจสูง สาเหตุคือโมเดลไม่เคยเห็นภาพที่ไม่ใช่ใบหน้าเลยและไม่มีคลาสพื้นหลัง (Background) จึงต้องเลือกหนึ่งในสามคลาสเสมอ ประกอบกับข้อมูลเอนไปทางคลาส with_mask สคริปต์ของอาจารย์เองก็ระบุไว้ว่าเป็น Sliding-window region classification demo ไม่ใช่ Fast R-CNN detection จริง ตัวเลขใน Confusion Matrix ที่ดูดีจึงไม่ได้สะท้อนความสามารถในการหาตำแหน่งวัตถุ และไม่มีค่า mAP

### สรุป

Fast R-CNN ให้ผลสูงกว่า R-CNN เล็กน้อยทั้ง Accuracy (92.74% เทียบกับ 90.38%) และ Macro F1 (0.63 เทียบกับ 0.57) โดยความต่างหลักมาจากความแม่นยำของคลาส without_mask อย่างไรก็ตามทั้งสองโมเดลเทรนเพียงครั้งเดียว และคลาสที่มีข้อมูลน้อย (16 ตัวอย่าง) ทำให้ตัวเลขแกว่งได้ง่ายระหว่างการเทรนแต่ละรอบ จึงสรุปได้เพียงว่าสองโมเดลให้ผลใกล้เคียงกัน โดย Fast R-CNN ดีกว่าเล็กน้อยในการทดลองครั้งนี้ ส่วนแนวทางพัฒนาต่อ ได้แก่ เพิ่มคลาส Background แล้วเทรนใหม่เพื่อลดกรอบที่เกิน เพิ่มข้อมูลหรือถ่วงน้ำหนักคลาส mask_weared_incorrect และเพิ่ม Region Proposal กับ NMS เพื่อให้เป็นการตรวจจับวัตถุจริงและวัดผลด้วย mAP

---

## 🚀 How to Use

```bash
# 1. เตรียม environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. ดาวน์โหลดชุดข้อมูลและแปลง label เป็น YOLO format
pip install kaggle
kaggle datasets download -d andrewmvd/face-mask-detection --unzip
python convert.py

# 3. เทรน ประเมินผล และทดสอบ (รันในโฟลเดอร์ 01-R-CNN หรือ 02-Fast-R-CNN)
cd 01-R-CNN
python train.py --limit 0 --epochs 10
python evaluate.py --split test
python image_test.py
python realtime_test.py --camera 0 --threshold 0.95 --regions 25   # กด Q เพื่อออก
```

ค่า `--limit 0` หมายถึงใช้ข้อมูล Train ทั้งหมด การดาวน์โหลดด้วย `kaggle` ต้องตั้งค่า API token (`kaggle.json`) ไว้ก่อน หรือจะดาวน์โหลดจากหน้าเว็บ Kaggle แล้วแตก zip ไว้ข้างๆ `convert.py` ก็ได้ ต้องได้โฟลเดอร์ `images/` และ `annotations/` ก่อนรัน `convert.py`

---

## 📁 Project Structure

```text
ML-09-Object_Detection_I/
├── 01-R-CNN/
│   ├── _core.py                 # ฟังก์ชันโหลดโมเดล/ตรวจจับ/วาดกรอบ สำหรับ realtime_test.py
│   ├── evaluate.py              # ประเมินโมเดลบนชุดทดสอบ (Accuracy, Precision, Recall, F1)
│   ├── image_test.py            # ทดสอบกับภาพนิ่งและบันทึกผลลัพธ์
│   ├── realtime_test.py         # ทดสอบแบบเรียลไทม์ผ่านเว็บแคม
│   ├── train.py                 # เทรนโมเดลและบันทึกเป็น model.keras
│   └── outputs/                 # ผลลัพธ์ (โมเดล, ภาพผลทดสอบ)
├── 02-Fast-R-CNN/
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   ├── train.py
│   └── outputs/
├── dataset/                     # สร้างโดย convert.py (YOLO format)
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
├── convert.py                   # แปลง label จาก XML (Pascal VOC) เป็น YOLO และแบ่ง Train / Val / Test
└── requirements.txt
```
