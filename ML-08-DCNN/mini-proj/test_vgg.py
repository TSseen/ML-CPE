import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing import image

def test_4_classes_predictions(model_path, img_paths, class_names):
    print("กำลังโหลดโมเดล...")
    model = tf.keras.models.load_model(model_path)
    
    plt.figure(figsize=(10, 10))
    
    for i, img_path in enumerate(img_paths):
        true_class = os.path.basename(os.path.dirname(img_path))
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        predictions = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = 100 * np.max(predictions[0])
        
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        
        text_color = 'green' if true_class == predicted_class else 'red'
        
        # แสดงผลให้เห็นว่าเฉลยคืออะไร ทายว่าอะไร
        plt.title(f"True: {true_class}\nPred: {predicted_class}\nConf: {confidence:.2f}%", 
                  color=text_color, fontsize=11)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('outputs/prediction_4_classes.png')
    plt.show()

if __name__ == "__main__":
    MODEL_PATH = "outputs/MobileNetV2_trained_model.h5" 
    VAL_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
    
    # 4 คลาสตามที่เทรน
    CLASS_NAMES = [
        'Apple___Apple_scab', 
        'Apple___Black_rot', 
        'Apple___Cedar_apple_rust', 
        'Apple___healthy'
    ]
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VAL_DIR):
        selected_imgs = []
        
        # สุ่มหยิบรูปมาคลาสละ 1 รูป
        for class_name in CLASS_NAMES:
            class_folder = os.path.join(VAL_DIR, class_name)
            if os.path.exists(class_folder):
                images_in_class = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images_in_class:
                    random_img = random.choice(images_in_class)
                    selected_imgs.append(os.path.join(class_folder, random_img))
        
        # ตรวจสอบว่าได้มาครบ 4 รูปหรือไม่
        if len(selected_imgs) == 4:
            print("สุ่มรูปมาคลาสละ 1 รูปเรียบร้อยครับ...")
            test_4_classes_predictions(MODEL_PATH, selected_imgs, CLASS_NAMES)
        else:
            print("หาไฟล์รูปภาพได้ไม่ครบทั้ง 4 คลาสครับ")
    else:
        print("ตรวจสอบ Path ของโมเดลหรือโฟลเดอร์รูปภาพอีกครั้ง")