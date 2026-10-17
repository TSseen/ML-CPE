😷 ML-09: Object Detection I

Python TensorFlow Status

LAB 09 ของรายวิชา Machine Learning ว่าด้วยการทดลองสถาปัตยกรรม R-CNN และ Fast R-CNN แบบง่าย (simplified educational implementation) ด้วย Python และ TensorFlow/Keras โค้ดต้นแบบมาจากอาจารย์ ซึ่งใช้ชุดข้อมูล BDD100K ส่วนงานนี้เปลี่ยนมาใช้ชุดข้อมูล Face Mask Detection จาก Kaggle เพื่อตรวจจับว่าบุคคลในภาพสวมหน้ากากอนามัยหรือไม่

📊 ชุดข้อมูล (Dataset)

ใช้ชุดข้อมูล Face Mask Detection (ผู้จัดทำ: andrewmvd, สัญญาอนุญาต CC0-1.0) มีภาพทั้งหมด 853 ภาพ พร้อมกรอบวัตถุ (bounding box) ในรูปแบบ Pascal VOC (XML) ซึ่งสคริปต์ convert.py จะแปลงเป็น YOLO format และแบ่งข้อมูลเป็นชุด train / val / test ในสัดส่วน 70 / 15 / 15 โดยสุ่มแบบกำหนด seed ไว้ที่ 42 เพื่อให้ได้ผลซ้ำเดิมทุกครั้ง

Class ID	ชื่อคลาส	ความหมาย
0	with_mask	สวมหน้ากากถูกต้อง
1	without_mask	ไม่สวมหน้ากาก
2	mask_weared_incorrect	สวมหน้ากากไม่ถูกต้อง

ชุดข้อมูลมีความไม่สมดุลสูง คลาส with_mask มีมากที่สุด (ในชุดทดสอบ 461 จาก 551 crops หรือประมาณ 84%) ส่วนคลาส mask_weared_incorrect มีเพียง 16 crops ซึ่งส่งผลต่อการวิเคราะห์ผลด้านล่าง

📂 โครงสร้างโฟลเดอร์ (Project Structure)
text
ML-09-Object_Detection_I/
├── dataset/                  # สร้างโดย convert.py (YOLO format)
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   ├── test/{images,labels}/
│   └── data.yaml
├── 01-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   ├── _core.py              # ฟังก์ชันสำหรับ realtime_test.py
│   └── outputs/
├── 02-Fast-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
├── convert.py                # แปลง Pascal VOC (XML) เป็น YOLO + แบ่ง train/val/test
├── requirements.txt
└── README.md
ไฟล์	หน้าที่
convert.py	แปลง label จาก XML เป็น YOLO format และสร้างโฟลเดอร์ dataset/
train.py	เทรนโมเดลและบันทึกเป็น outputs/model.keras
evaluate.py	ประเมินผลบนชุดทดสอบ (accuracy, precision, recall, F1)
image_test.py	ทดสอบกับภาพนิ่งและบันทึกผลลง outputs/
realtime_test.py	ทดสอบแบบเรียลไทม์ผ่านเว็บแคม กด Q เพื่อออก
🚀 วิธีการใช้งาน (How to Use)

1. เตรียม environment

bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2. ดาวน์โหลดชุดข้อมูลและแปลง label

bash
pip install kaggle
kaggle datasets download -d andrewmvd/face-mask-detection --unzip
python convert.py

ขั้นตอนนี้ต้องตั้งค่า Kaggle API token ไว้ก่อน (ไฟล์ kaggle.json) หรือจะดาวน์โหลดจากหน้าเว็บ Kaggle แล้วแตก zip ไว้ในโฟลเดอร์เดียวกับ convert.py ก็ได้ ต้องได้โฟลเดอร์ images/ และ annotations/ ก่อนรัน convert.py

3. เทรน ประเมินผล และทดสอบ (รันในโฟลเดอร์ของแต่ละสถาปัตยกรรม)

bash
cd 01-R-CNN          # หรือ 02-Fast-R-CNN
python train.py --limit 0 --epochs 10
python evaluate.py --split test
python image_test.py
python realtime_test.py --camera 0 --threshold 0.95 --regions 25

ค่า --limit 0 หมายถึงใช้ข้อมูล train ทั้งหมด ส่วน --limit 200 จะใช้เพียง 200 ภาพแรก ซึ่งเทรนเร็วกว่าแต่ผลไม่เต็มที่

🧪 ผลการทดลอง (Results)

ประเมินบนชุดทดสอบจำนวน 551 crops (ตัดจากกรอบจริงใน label) ทั้งสองโมเดลเทรนด้วยข้อมูล train เต็มชุดและจำนวน epoch เท่ากัน

โมเดล	Accuracy	Macro Precision	Macro Recall	Macro F1
R-CNN	0.904	0.55	0.59	0.57
Fast R-CNN	0.927	0.73	0.60	0.63

รายละเอียดแยกตามคลาส (Precision / Recall / F1)

คลาส	จำนวน (support)	R-CNN	Fast R-CNN
with_mask	461	0.94 / 0.95 / 0.94	0.93 / 0.99 / 0.96
without_mask	74	0.70 / 0.84 / 0.76	0.92 / 0.74 / 0.82
mask_weared_incorrect	16	0.00 / 0.00 / 0.00	0.33 / 0.06 / 0.11
💡 สรุปและข้อสังเกต (Discussion)

Fast R-CNN ให้ผลสูงกว่า R-CNN เล็กน้อยทั้ง accuracy (0.927 เทียบกับ 0.904) และ macro F1 (0.63 เทียบกับ 0.57) ทั้งสองโมเดลจำแนกคลาส with_mask ได้ดีมากเพราะเป็นคลาสที่มีข้อมูลมากที่สุด ส่วนความต่างหลักอยู่ที่คลาส without_mask ซึ่ง Fast R-CNN ทายแม่นกว่าชัดเจน (precision 0.92 เทียบกับ 0.70) แต่จับได้ไม่ครบเท่า R-CNN (recall 0.74 เทียบกับ 0.84) ส่วนคลาส mask_weared_incorrect ทั้งสองโมเดลทำได้ต่ำมาก โดย R-CNN ทายถูกไม่ได้เลย

ควรอ่านตัวเลขเหล่านี้อย่างระมัดระวัง เพราะค่า accuracy ที่สูงมาจากข้อมูลที่เอนไปทางคลาส with_mask ค่า macro F1 สะท้อนผลจริงได้ดีกว่า นอกจากนี้ทั้งสองโมเดลเทรนเพียงครั้งเดียว และคลาสที่มีข้อมูลน้อย (16 ตัวอย่าง) ทำให้ตัวเลขแกว่งได้ง่ายระหว่างการเทรนแต่ละรอบ จึงสรุปได้เพียงว่าสองโมเดลให้ผลใกล้เคียงกัน โดย Fast R-CNN ดีกว่าเล็กน้อยในการทดลองครั้งนี้

⚠️ ข้อจำกัด (Limitations)

โค้ดของทั้งสองโมเดลเป็นเพียง crop-classification baseline ไม่ใช่ R-CNN หรือ Fast R-CNN ฉบับสมบูรณ์ เพราะไม่มี region proposal, bounding-box regression และ NMS ผลที่วัดได้จึงเป็นความสามารถในการจำแนกภาพที่ตัดมาจากกรอบจริงเท่านั้น ไม่มีค่า mAP สำหรับวัดการหาตำแหน่งวัตถุ นอกจากนี้โมเดลไม่มีคลาสพื้นหลัง (background) เมื่อทดสอบกับเว็บแคมด้วยวิธี sliding window จึงเกิดกรอบเกินจำนวนมาก เพราะทุกหน้าต่างที่ตกบนผนังหรือเสื้อก็ถูกบังคับให้ทายเป็นหนึ่งในสามคลาส

🔮 แนวทางพัฒนาต่อ
เพิ่มคลาส background และเทรนใหม่ เพื่อลดกรอบที่เกิน
เพิ่มข้อมูลของคลาส mask_weared_incorrect หรือถ่วงน้ำหนักคลาส (class weights)
เพิ่มขั้นตอน region proposal และ NMS เพื่อให้เป็นการตรวจจับวัตถุจริง และวัดผลด้วย mAP
เทรนซ้ำหลายรอบแล้วเฉลี่ยผล เพื่อให้การเปรียบเทียบน่าเชื่อถือขึ้น
📚 อ้างอิง (References)
Face Mask Detection (Kaggle, andrewmvd)
R-CNN และ Fast R-CNN
ต้นแบบโค้ดและโครงสร้างโปรเจกต์จาก aproot-en/Machine-Learning-Course
