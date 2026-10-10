import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing import image

def predict_random_image(model_path, val_dir, class_names):
    print(f"กำลังโหลดโมเดลจาก: {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    
    # 1. รวบรวมไฟล์รูปภาพทั้งหมดที่อยู่ในโฟลเดอร์คลาสที่เราเทรน
    all_image_paths = []
    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # เช็คว่ารูปนี้อยู่ในคลาสที่เราเทรนหรือไม่
                for class_name in class_names:
                    if class_name in root:
                        all_image_paths.append(os.path.join(root, file))
                        break

    if not all_image_paths:
        print("ไม่พบรูปภาพในโฟลเดอร์ Validation เลยครับ")
        return

    # 2. สุ่มเลือกรูปมา 1 รูปจากทั้งหมด
    random_img_path = random.choice(all_image_paths)
    
    # ดึงชื่อคลาสที่แท้จริง (เฉลย) จากชื่อโฟลเดอร์ของรูปนั้น
    true_class = os.path.basename(os.path.dirname(random_img_path))
    
    print(f"\nสุ่มได้รูปภาพเฉลย (True Label): {true_class}")
    
    # 3. เตรียมรูปภาพเพื่อเข้าโมเดล
    img = image.load_img(random_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0 

    # 4. ให้โมเดลทำนายผล
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = 100 * np.max(predictions[0])
    
    # 5. แสดงผลลัพธ์
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    
    # กำหนดสี: ทายถูกเป็นสีเขียว ทายผิดเป็นสีแดง
    color = 'green' if true_class == predicted_class else 'red'
    
    plt.title(f"True: {true_class}\nPred: {predicted_class} ({confidence:.2f}%)", color=color, fontsize=14)
    plt.axis("off")
    
    output_img_path = 'outputs/random_prediction_result.png'
    plt.savefig(output_img_path)
    plt.show()

if __name__ == "__main__":
    # --- 1. ระบุไฟล์โมเดลที่คุณเทรนเสร็จแล้ว ---
    MODEL_PATH = "outputs/MobileNetV2_trained_model.h5" 
    
    # --- 2. โฟลเดอร์ Validation ที่เก็บรูปภาพแยกตามคลาส ---
    VAL_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
    
    # --- 3. ชื่อคลาสที่ใช้เทรน ---
    CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___healthy']
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VAL_DIR):
        predict_random_image(MODEL_PATH, VAL_DIR, CLASS_NAMES)
    else:
        print("ไม่พบไฟล์โมเดล หรือ โฟลเดอร์รูปภาพ กรุณาตรวจสอบ Path อีกครั้งครับ")