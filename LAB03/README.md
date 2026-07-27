# 📈 LAB 03: Regression & Classification

**รายวิชา:** Machine Learning (Sec 2, 1/2569)  
**ผู้จัดทำ:** Thirawat Saengklin (GitHub: [@TSseen](https://github.com/TSseen))  
**สถาบัน:** มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT)  

---

โปรเจกต์นี้จัดทำขึ้นเพื่อการศึกษาและเรียนรู้เกี่ยวกับการสร้างแบบจำลอง **Regression** และ **Classification** ซึ่งเป็นเทคนิคพื้นฐานของ Supervised Learning[cite: 1] รวมถึงการเตรียมข้อมูล การประเมินประสิทธิภาพ และการเปรียบเทียบผลลัพธ์ของโมเดลเพื่อนำไปประยุกต์ใช้งานจริง

## 🎯 วัตถุประสงค์ของใบงาน (Project Objectives)

1. **เข้าใจหลักการของ Regression และ Classification:** ซึ่งเป็นเทคนิคพื้นฐานของ Supervised Learning และสามารถอธิบายความแตกต่างระหว่างการทำนายค่าต่อเนื่อง (Continuous Value) และการจำแนกประเภทข้อมูลได้[cite: 1]
2. **เตรียมข้อมูลสำหรับการสร้างแบบจำลอง:** โดยเลือกใช้คุณลักษณะ (Features) ที่เหมาะสม รวมถึงประยุกต์ใช้เทคนิค Principal Component Analysis (PCA) เพื่อลดจำนวนคุณลักษณะและเพิ่มประสิทธิภาพของการเรียนรู้[cite: 1]
3. **พัฒนาแบบจำลองเชิงทำนาย:** สำหรับการทำนายอายุจากภาพใบหน้า (Linear Regression) และแบบจำลอง Classification สำหรับการจำแนกเพศจากภาพใบหน้า พร้อมเปรียบเทียบผลลัพธ์ของแต่ละวิธี[cite: 1]
4. **พัฒนาทักษะการโปรแกรมด้วยภาษา Python:** สามารถใช้ไลบรารีด้าน Machine Learning ในการสร้าง ฝึกสอน (Training) ทดสอบ (Testing) และประเมินประสิทธิภาพของแบบจำลอง[cite: 1]
5. **วิเคราะห์และอธิบายผลลัพธ์ของแบบจำลอง:** ด้วยตัวชี้วัดที่เหมาะสม เช่น Accuracy, Precision, Recall, F1-score, ROC Curve และ AUC รวมถึงนำเสนอผลงานผ่าน GitHub เพื่อจัดทำ Portfolio และเผยแพร่ผลงานทางวิชาการต่อไป[cite: 1]

---

## 🛠️ ขั้นตอนการทดลอง

### 📈 Part 1: Regression (การทำนายค่าต่อเนื่อง)[cite: 1]
* ศึกษาและสร้างแบบจำลอง **Simple Linear Regression** และ **Multiple Linear Regression**[cite: 1]
* ประยุกต์ใช้โมเดลสำหรับการทำนายอายุจากภาพใบหน้า (**Age Prediction**)[cite: 1]

### 🗂️ Part 2: Classification (การจำแนกประเภทข้อมูล)[cite: 1]
* เตรียมข้อมูลสำหรับการทำ Classification (**Preparing Classification Data**)[cite: 1]
* สร้างแบบจำลอง **Logistic Regression** เพื่อจำแนกเพศจากภาพใบหน้า (**Gender Prediction**)[cite: 1]
* แสดงผลการตัดสินใจของโมเดลด้วย **Decision Boundary Visualization** และประเมินผลด้วย **Confusion Matrix**[cite: 1]

### ⚖️ Part 3: Model Comparison (การเปรียบเทียบแบบจำลอง)[cite: 1]
* เปรียบเทียบประสิทธิภาพระหว่าง **Simple vs Multiple Linear Regression**[cite: 1]
* วิเคราะห์ความแตกต่างของผลลัพธ์ระหว่างข้อมูลฝึกสอนและข้อมูลทดสอบ (**Training vs Testing Performance**)[cite: 1]
* เปรียบเทียบความแตกต่างระหว่างการทำ **Regression vs Classification**[cite: 1]
* ประเมินผลลัพธ์รวมด้วยตัวชี้วัดประสิทธิภาพ (**Model Performance Metrics**)[cite: 1]

---

💡 *หมายเหตุ: โปรเจกต์นี้เป็นส่วนหนึ่งของการเรียนวิชา Machine Learning ประจำภาคเรียนที่ 1/2569 เท่านั้น*
