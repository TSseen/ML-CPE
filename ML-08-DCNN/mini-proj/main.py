import os
from data_loader import load_datasets
from preprocessing import apply_preprocessing
from models import build_mobilenet, build_vgg16, build_resnet50
from evaluate import plot_model_comparison

# --- Path ของ Dataset ---
TRAIN_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VAL_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 

# ใช้ 4 คลาสตามเดิม
SELECTED_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy'
]

def main():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    train_ds, val_ds, class_names = load_datasets(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, SELECTED_CLASSES)
    num_classes = len(class_names)
    
    train_ds, val_ds = apply_preprocessing(train_ds, val_ds)

    # รวม 3 โมเดลไว้ด้วยกันเพื่อวนลูปเทรน
    models_to_train = {
        "MobileNetV2": build_mobilenet(num_classes, IMG_SIZE),
        "VGG16": build_vgg16(num_classes, IMG_SIZE),
        "ResNet50": build_resnet50(num_classes, IMG_SIZE)
    }

    histories = {}

    for name, model in models_to_train.items():
        print(f"\n{'='*40}")
        print(f"กำลังเทรนโมเดล: {name}")
        print(f"{'='*40}")
        
        # เทรนโมเดล
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        
        # บันทึกประวัติและเซฟโมเดล
        histories[name] = history
        model.save(f"outputs/{name}_trained_model.h5")

    print("\n--- สร้างกราฟเปรียบเทียบ ---")
    plot_model_comparison(histories)

if __name__ == "__main__":
    main()