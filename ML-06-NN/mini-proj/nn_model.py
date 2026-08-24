import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 1. นำเข้าฟังก์ชันของอาจารย์ (สมมติเซฟไว้ชื่อ model_from_teacher.py หรือวางรวมไว้ในไฟล์เดียวกัน)
# แต่ในที่นี้ผมรวมฟังก์ชันของอาจารย์ไว้ให้ครบในไฟล์เดียวเพื่อให้รันง่ายครับ:
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes):
    """Fully-connected neural network (MLP)."""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax"
        ),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(X_train, y_train, X_val, y_val, num_classes, output_dir=None, epochs=80, batch_size=32):
    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "nn_model.keras"))
        print(f"Saved: {os.path.join(output_dir, 'nn_model.keras')}")

    return model, history


# 2. ฟังก์ชันโหลด Dataset สำหรับ FER-2013 (ใช้ PIL ป้องกันปัญหา Path ภาษาไทย/OneDrive)
def load_fer2013_data(data_dir, img_size=(48, 48)):
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
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                img_array = np.array(img, dtype='float32')
                # หมายเหตุ: โค้ดของอาจารย์มี layers.Rescaling(1.0 / 255) อยู่แล้ว 
                # ส่งค่าดิบ 0-255 เข้าไปได้เลย หรือจะหาร 255 ตรงนี้เลยก็ได้ครับ
                images.append(img_array)
                labels.append(class_to_idx[cls_name])
            except Exception:
                pass
                
    X = np.array(images, dtype='float32')
    y = np.array(labels)
    return X, y, class_to_idx


# 3. จุดเริ่มต้นการรันโปรแกรม
if __name__ == "__main__":
    # ใส่ Path โฟลเดอร์ train ของคุณตรงนี้ (แนะนำให้ย้ายมาไว้ที่ C:\dataset\train ถ้าติดปัญหา OneDrive)
    data_dir = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\Machine-Learning-Course\ML-06-NN\mini-proj\train"
    
    X, y, class_to_idx = load_fer2013_data(data_dir)
    print(f"โหลดข้อมูลสำเร็จ! จำนวน: {X.shape[0]} รูป | คลาส: {class_to_idx}")
    
    # แบ่ง Train / Validation Set (80:20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # รันฟังก์ชันเทรนของอาจารย์
    num_classes = len(class_to_idx)
    model, history = train_model(
        X_train, y_train, X_val, y_val, 
        num_classes=num_classes, 
        output_dir="models", 
        epochs=80, 
        batch_size=64
    )
def predict_model(model, X_test):
    """Predict class labels for the test set."""
    probabilities = model.predict(X_test, verbose=0)
    
    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)
        
    return probabilities.argmax(axis=1)