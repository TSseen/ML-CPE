import os
import cv2
import numpy as np
from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path, img_size=48, max_items=None):
    images, labels_gender, labels_age = [], [], []
    filenames = sorted([f for f in os.listdir(data_path) if f.lower().endswith(VALID_EXT)])
    
    loaded = 0
    for filename in filenames:
        if max_items and loaded >= max_items: break
        
        parts = filename.split('_')
        if len(parts) < 3: continue
            
        try:
            age = int(parts[0])      # ดึงอายุจากส่วนแรกของชื่อไฟล์
            gender = int(parts[1])   # ดึงเพศจากส่วนที่สอง
        except ValueError:
            continue

        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path)
        if image is None: continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = preprocess_image(image, img_size)

        images.append(image)
        labels_gender.append(gender)
        labels_age.append(age)
        loaded += 1

    print(f"โหลดสำเร็จ {loaded} รูปภาพ")
    return np.stack(images), np.array(labels_gender), np.array(labels_age)