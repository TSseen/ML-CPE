import os
import tensorflow as tf
from data_loader import load_datasets
from preprocessing import apply_preprocessing
from evaluate import plot_and_save_confusion_matrix

# --- Path ของ Dataset ---
TRAIN_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VAL_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ต้องใช้คลาสให้ตรงกับตอนที่คุณเทรนโมเดลไป
SELECTED_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy'
]

def generate_cm_only():
    print("กำลังเตรียมชุดข้อมูล Validation...")
    # โหลดแค่ validation เพื่อมาทดสอบ ไม่ต้องใช้ train
    _, val_ds, class_names = load_datasets(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, SELECTED_CLASSES)
    
    # ปรับสเกลข้อมูลให้ตรงกับที่โมเดลต้องการ
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # รายชื่อโมเดลที่เราคาดว่าเซฟไว้แล้วในโฟลเดอร์ outputs
    saved_models = [
        "MobileNetV2_trained_model.h5",
        "VGG16_trained_model.h5",
        "ResNet50_trained_model.h5"
    ]
    
    for model_file in saved_models:
        model_path = os.path.join("outputs", model_file)
        
        if os.path.exists(model_path):
            print(f"\n{'='*40}")
            print(f"พบไฟล์โมเดล: {model_file} (ไม่ต้องเทรนใหม่!)")
            print(f"{'='*40}")
            
            # โหลดโมเดลที่เทรนเสร็จแล้วขึ้นมา
            model = tf.keras.models.load_model(model_path)
            
            # ดึงชื่อโมเดลมาตั้งชื่อไฟล์รูป (ตัด .h5 ออก)
            model_name = model_file.replace("_trained_model.h5", "")
            
            # สั่งวาด Confusion Matrix ทันที
            plot_and_save_confusion_matrix(model, val_ds, class_names, model_name=model_name)
            
        else:
            print(f"\n[ข้าม] ไม่พบไฟล์ {model_file} (อาจจะยังเทรนไม่เสร็จ หรือไม่ได้เทรน)")

if __name__ == "__main__":
    generate_cm_only()