import os
import numpy as np
from PIL import Image

def load_fer2013_data(data_dir, img_size=(48, 48)):
    """โหลดข้อมูลภาพจากโฟลเดอร์โดยใช้ PIL เพื่อรองรับ Path ภาษาไทยบน Windows"""
    images = []
    labels = []
    
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"กำลังโหลดข้อมูลจาก: {data_dir}")
    for cls_name in classes:
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
            
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                # ใช้ PIL โหลดภาพและแปลงเป็น RGB เพื่อให้เข้ากับ preprocessing.py
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img, dtype='uint8')
                
                images.append(img_array)
                labels.append(class_to_idx[cls_name])
            except Exception:
                pass
                
    X = np.array(images, dtype='uint8')
    y = np.array(labels)
    
    return X, y, class_to_idx

# Alias สำรองเพื่อให้รองรับชื่อฟังก์ชันเดิม
load_data = load_fer2013_data