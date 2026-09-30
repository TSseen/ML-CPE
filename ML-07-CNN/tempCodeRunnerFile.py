import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import load_data
from cnn_model import build_model

def main():
    dataset_path = "utkface_aligned_cropped/UTKFace" 
    
    print("1. กำลังโหลดข้อมูลรูปภาพ...")
    # โหลดทั้งเพศและอายุ
    X, y_gender, y_age = load_data(dataset_path, img_size=48) 
    
    print("\n2. กำลังแบ่งข้อมูล Train / Test...")
    X_train, X_test, y_g_train, y_g_test, y_a_train, y_a_test = train_test_split(
        X, y_gender, y_age, test_size=0.2, random_state=42
    )

    print("\n3. กำลังสร้างและคอมไพล์โมเดล...")
    model = build_model(input_shape=(48, 48, 3)) 
    
    # คอมไพล์โมเดลแบบ 2 หัว
    model.compile(
        optimizer='adam',
        loss={'gender_out': 'binary_crossentropy', 'age_out': 'mse'},
        metrics={'gender_out': 'accuracy', 'age_out': 'mae'} # mae = Mean Absolute Error (ความคลาดเคลื่อนของอายุโดยเฉลี่ย)
    )
    
    print("\n4. เริ่มขั้นตอนการ Training...")
    history = model.fit(
        X_train, 
        {'gender_out': y_g_train, 'age_out': y_a_train},
        validation_data=(X_test, {'gender_out': y_g_test, 'age_out': y_a_test}),
        epochs=50, 
        batch_size=64
    )
    
    os.makedirs("outputs", exist_ok=True)
    model.save("outputs/gender_age_model.h5")
    print("\n[สำเร็จ] บันทึกโมเดลไว้ที่: outputs/gender_age_model.h5")

if __name__ == "__main__":
    main()