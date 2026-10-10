# ML-08-Deep Convolutional Neural Network (DCNN)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Library](https://img.shields.io/badge/Library-PyTorch%20%2F%20TensorFlow-orange.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## Dataset

**New Plant Diseases Dataset (Augmented)** : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

ชุดข้อมูลภาพใบแอปเปิลสำหรับจำแนกโรค แบ่งเป็น 4 คลาส  ได้แก่ Apple scab, Black rot, Cedar apple rust และ Healthy ชุดทดสอบมีทั้งหมด 1,943 รูป (504, 497, 440 และ 502 รูปตามลำดับ) ซึ่งกระจายตัวค่อนข้างสมดุล จึงใช้ Accuracy เป็นตัวชี้วัดหลักได้อย่างเหมาะสม

---

## 📌 Project Overview

โปรเจกต์นี้เปรียบเทียบประสิทธิภาพของ Deep Convolutional Neural Network (DCNN) แบบ Transfer Learning 3 สถาปัตยกรรม คือ MobileNetV2, VGG16 และ ResNet50 ในการจำแนกโรคจากภาพใบแอปเปิล โดยใช้โมเดลที่ผ่านการเทรนบน ImageNet มาก่อนแล้วนำมาปรับให้เข้ากับงานนี้ เทรนทุกโมเดลภายใต้เงื่อนไขเดียวกันเป็นเวลา 10 Epochs แล้วนำผลมาเทียบกันผ่านกราฟ Training/Validation Accuracy และ Confusion Matrix

---

## 📊 ผลการทดลองและการวิเคราะห์

### เปรียบเทียบภาพรวม

| โมเดล | Training Acc | Validation Acc |
|---|---|---|
| MobileNetV2 | 99.33% | 99.28% |
| VGG16 | 96.87% | 97.53% |
| ResNet50 | 63.61% | 66.80% |

### MobileNetV2

MobileNetV2 เป็นโมเดลที่ดีที่สุดในการทดลองนี้ จากกราฟ Accuracy จะเห็นว่าตั้งแต่ Epoch แรกโมเดลมี Training Accuracy สูงถึง 92% และ Validation Accuracy 98% แล้ว จากนั้นไต่ขึ้นไปนิ่งที่ประมาณ 99% ตั้งแต่ Epoch ที่ 2 เป็นต้นไป เส้น Training และ Validation อยู่ชิดกันตลอดจนจบที่ 99.33% และ 99.28% แสดงว่าไม่เกิด Overfitting มีจุดที่น่าสังเกตเพียงจุดเดียวคือ Validation Accuracy ตกลงเหลือราว 96.8% ที่ Epoch 8 แต่กลับขึ้นมาที่ 99.28% ใน Epoch ถัดไป ซึ่งเป็นความผันผวนเล็กน้อยระหว่างการเทรน ไม่ได้กระทบผลสุดท้าย

ใน Confusion Matrix จาก 1,943 รูป ทำนายถูก 1,929 รูป ผิดเพียง 14 รูป ความแม่นยำรายคลาสสูงทุกคลาส คือ Healthy 99.8%, Black rot 99.6%, Cedar apple rust 99.5% และ Apple scab 98.2% ความผิดพลาดส่วนใหญ่คือการที่ Apple scab ถูกทำนายเป็น Black rot 5 รูป ซึ่งถือว่าน้อยมาก

### VGG16

VGG16 ให้ผลดีเป็นอันดับสอง กราฟแสดงว่าโมเดลเริ่มต้นช้ากว่า MobileNetV2 อย่างชัดเจน โดย Training Accuracy ใน Epoch แรกอยู่ที่ประมาณ 74% แต่เรียนรู้ได้เร็วและเข้าใกล้ 95% ภายใน Epoch ที่ 3 ถึง 4 จากนั้นค่อยๆ เพิ่มจนจบที่ 96.87% ส่วน Validation Accuracy จบที่ 97.53% ซึ่งสูงกว่า Training เล็กน้อย เป็นลักษณะที่พบได้เมื่อใช้ Data Augmentation หรือ Dropout ตอนเทรน และบ่งชี้ว่าโมเดลไม่ Overfit เส้นกราฟยังเพิ่มขึ้นช้าๆ จึงน่าจะดีขึ้นได้อีกหากเทรนนานกว่านี้

ใน Confusion Matrix ทำนายถูก 1,895 รูป ผิด 48 รูป Healthy (99.4%) และ Black rot (99.2%) แม่นยำมาก ส่วนจุดอ่อนอยู่ที่ Apple scab (95.0%) ซึ่งถูกทำนายผิดเป็น Healthy 14 รูป และ Cedar apple rust (96.4%) ที่ถูกทำนายผิดกระจายไปคลาสอื่นๆ โดยรวมความผิดพลาดกระจายตัวและมีจำนวนน้อย

### ResNet50

ResNet50 ให้ผลแย่ที่สุดอย่างชัดเจน จากกราฟ Training Accuracy เริ่มที่ประมาณ 34% และ Validation เริ่มที่ 49% จากนั้นไต่ขึ้นอย่างช้าๆ จนจบที่ 63.61% และ 66.80% ตามลำดับ จุดสำคัญคือเส้นกราฟยังชันขึ้นอยู่ตอนจบ Epoch ที่ 10 หมายความว่าโมเดลยังไม่ลู่เข้า (Underfitting) ไม่ใช่ว่าเรียนรู้เต็มที่แล้ว การที่ Validation สูงกว่า Training เล็กน้อยก็เป็นสัญญาณว่าโมเดลยังไม่ได้เรียนรู้ข้อมูลได้เต็มที่เช่นกัน

Confusion Matrix สะท้อนปัญหานี้ชัดเจน ทำนายถูกเพียง 1,298 รูปจาก 1,943 รูป คลาส Apple scab แย่ที่สุด ถูกต้องเพียง 99 จาก 504 รูป (19.6%) โดยถูกทำนายผิดเป็น Cedar apple rust ถึง 225 รูป และเป็น Black rot 113 รูป ส่วนคลาส Healthy ก็สับสนกับ Cedar apple rust 71 รูป (Recall 75.9%) ขณะที่ Black rot (86.5%) และ Cedar apple rust (88.2%) ทำได้ดีกว่าคลาสอื่น แต่ยังห่างจากอีกสองโมเดลมาก

สาเหตุที่เป็นไปได้ที่ควรตรวจสอบ ได้แก่ การทำ Preprocessing ไม่ตรงกับที่ ResNet50 ต้องการ (เช่น การ Normalize ที่ต่างจากตอนเทรน ImageNet), Learning Rate ที่ไม่เหมาะสม, การ Freeze หรือ Unfreeze Layer ที่ไม่เหมาะ และจำนวน Epoch ที่น้อยเกินไป ผลของ ResNet50 จึงไม่ควรตีความว่าสถาปัตยกรรมนี้แย่โดยตัวมันเอง แต่เป็นผลจากการตั้งค่าในการทดลองนี้

### สรุป

จากการเปรียบเทียบ DCNN ทั้งสามสถาปัตยกรรม MobileNetV2 ให้ความแม่นยำสูงสุดที่ 99.28% และเรียนรู้เร็วที่สุด อีกทั้งเป็นโมเดลขนาดเล็กที่เบาที่สุดในสามตัว จึงเหมาะที่สุดสำหรับนำไปใช้งานจริงหรือรันบนอุปกรณ์ที่มีทรัพยากรจำกัด VGG16 เป็นตัวเลือกที่ดีรองลงมาด้วยความแม่นยำ 97.53% แต่เป็นโมเดลที่ใหญ่และหนักกว่า ส่วน ResNet50 ต้องปรับการตั้งค่าและเทรนเพิ่มก่อนจึงจะเปรียบเทียบได้อย่างเป็นธรรม

---
## 📁 Project Structure

```text
ML-08-DCNN/
├── mini-proj/
│   ├── data_loader.py           # โหลด Dataset และสร้าง Data Batches
│   ├── evaluate.py              # ประเมินโมเดล (Accuracy)
│   ├── generate_cm.py           # สร้างภาพ Confusion Matrix ของแต่ละโมเดล
│   ├── main.py                  # รัน Training Pipeline
│   ├── models.py                # สร้างโมเดล MobileNetV2, VGG16 และ ResNet50
│   ├── predict.py               # ทำนายโรคจากรูปใบแอปเปิลใหม่ (Inference)
│   ├── preprocessing.py         # Image Preprocessing และ Data Augmentation
│   ├── split_data.py            # แบ่งข้อมูลเป็นชุด Train / Validation / Test
│   └── test_vgg.py              # ทดสอบโมเดล VGG16
└── outputs/                     # ผลลัพธ์ (กราฟ, Weights ของโมเดล, Logs)
```
