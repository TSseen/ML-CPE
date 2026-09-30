import cv2
import numpy as np

def preprocess_image(image, img_size):
    if image is None: return None
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    return image
    # 1. ปรับขนาดภาพให้อยู่ในสเกลเดียวกัน
    image = cv2.resize(image, (img_size, img_size))
    
    # 2. ทำ Normalization ปรับค่าพิกเซลจาก 0-255 ให้อยู่ในช่วง 0.0 - 1.0
    image = image / 255.0
    
    # 3. จัดการมิติภาพสำหรับเข้าโมเดล
    # (ถ้าโหลดภาพแบบ Grayscale มิติจะเป็น 2 มิติ ต้องเพิ่มแกนให้เป็น 3 มิติ)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        
    return image