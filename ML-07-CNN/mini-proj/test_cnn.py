import cv2
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model

def test_multiple_faces(image_list, model_path="outputs/gender_age_model.h5"):
    print(f"1. กำลังโหลดโมเดลจาก: {model_path}")
    try:
        model = load_model(model_path, compile=False) 
    except Exception as e:
        print(f"[ข้อผิดพลาด]: ไม่สามารถโหลดโมเดลได้ เพราะ -> {e}")
        return

    cascade_filename = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_filename):
        print("2. กำลังดาวน์โหลดไฟล์ตรวจจับใบหน้า...")
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, cascade_filename)
    
    face_cascade = cv2.CascadeClassifier(cascade_filename)
    os.makedirs("outputs", exist_ok=True) 

    for image_path in image_list:
        print(f"\n----------------------------------------")
        print(f"กำลังประมวลผลรูปภาพ: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ข้าม] หารูปภาพ '{image_path}' ไม่พบ")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ปรับ minSize=(80, 80) เพื่อกรองหน้าปลอมที่เล็กเกินไปทิ้ง
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))
        print(f"-> พบใบหน้าทั้งหมด {len(faces)} คน")

        for (x, y, w, h) in faces:
            face_crop_color = img[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            
            predictions = model.predict(face_input, verbose=0)
            
            if isinstance(predictions, list) and len(predictions) == 2:
                gender_pred = predictions[0][0][0]
                age_pred = predictions[1][0][0]
                label = "Female" if gender_pred > 0.5 else "Male"
                pred_age = int(age_pred)
                text = f"{label}, Age: {pred_age}"
            else:
                gender_pred = predictions[0][0]
                label = "Female" if gender_pred > 0.5 else "Male"
                text = f"{label} (No Age Data)"

            color = (147, 20, 255) if label == "Female" else (255, 144, 30) 
            
            # ปรับตัวหนังสือให้ใหญ่ขึ้น (1.2) และเส้นหนาขึ้น (3)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        filename = os.path.basename(image_path)
        save_path = os.path.join("outputs", f"predicted_{filename}")
        cv2.imwrite(save_path, img)
        print(f"[สำเร็จ] บันทึกรูปภาพไว้ที่: {save_path}")

        cv2.imshow(f"Result - {filename}", img)
        print(">>> กดปุ่มอะไรก็ได้บนคีย์บอร์ด (หรือกดกากบาท) เพื่อดูรูปถัดไป <<<")
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

if __name__ == "__main__":
    images_to_test = [
        "1.jpg",
        "3.jpg",
        "Blackpink.jpg",
        "5.jpg"  
    ]
    test_multiple_faces(images_to_test)