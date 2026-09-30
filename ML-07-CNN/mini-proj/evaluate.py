import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from data_loader import load_data

def evaluate_multi_task_model():
    model_path = "outputs/gender_age_model.h5"
    dataset_path = "utkface_aligned_cropped/UTKFace"
    
    print("1. กำลังโหลดโมเดลที่เซฟไว้...")
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        print(f"[ข้อผิดพลาด] ไม่สามารถโหลดโมเดลได้: {e}")
        return
    
    print("2. กำลังโหลดข้อมูลชุดทดสอบ...")
    # โหลดข้อมูล รูปภาพ, เพศ, อายุ
    X, y_gender, y_age = load_data(dataset_path, img_size=48)
    
    # ต้องแบ่งข้อมูลด้วย random_state=42 เหมือนตอน Training เพื่อเอาชุด Test มาใช้วัดผล
    _, X_test, _, y_g_test, _, y_a_test = train_test_split(
        X, y_gender, y_age, test_size=0.2, random_state=42
    )
    print(f"\nจำนวนรูปภาพที่ใช้ประเมินผล (Test Set): {len(X_test)} รูป")

    print("\n3. เริ่มประเมินผลการทำนาย (Predicting)...")
    predictions = model.predict(X_test)
    
    # แยกผลลัพธ์ออกเป็น 2 กิ่ง
    gender_preds = predictions[0].flatten()
    age_preds = predictions[1].flatten()

    # แปลงความน่าจะเป็นของเพศให้เป็น 0 หรือ 1 (0: Male, 1: Female)
    y_g_pred_classes = (gender_preds > 0.5).astype(int)
    
    print("\n==========================================")
    print("📊 ผลการประเมิน: การทายเพศ (Gender)")
    print("==========================================")
    target_names = ['Male', 'Female']
    print(classification_report(y_g_test, y_g_pred_classes, target_names=target_names))
    
    print("==========================================")
    print("📈 ผลการประเมิน: การทายอายุ (Age)")
    print("==========================================")
    mae = mean_absolute_error(y_a_test, age_preds)
    rmse = np.sqrt(mean_squared_error(y_a_test, age_preds))
    print(f"ความคลาดเคลื่อนเฉลี่ย (MAE): {mae:.2f} ปี")
    print(f"ความคลาดเคลื่อนรุนแรงเฉลี่ย (RMSE): {rmse:.2f} ปี")
    
    print("\n4. กำลังสร้างกราฟประเมินผลเชิงลึก...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # กราฟ 1: Confusion Matrix ของเพศ (ดูว่าทายชายเป็นหญิง หรือหญิงเป็นชายเยอะแค่ไหน)
    cm = confusion_matrix(y_g_test, y_g_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14})
    axes[0].set_title('Gender Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('Actual Label', fontsize=12)

    # กราฟ 2: Scatter Plot ของอายุ (ดูการกระจายตัวของความผิดพลาด)
    axes[1].scatter(y_a_test, age_preds, alpha=0.3, color='darkorange', edgecolor='white')
    axes[1].plot([0, 116], [0, 116], 'r--', linewidth=2, label='Perfect Prediction') # เส้นประสีแดงคือเส้นที่ทายถูก 100%
    axes[1].set_title(f'Age Prediction Performance\n(MAE: {mae:.2f} years)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual Age', fontsize=12)
    axes[1].set_ylabel('Predicted Age', fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/evaluation_multi_task.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[สำเร็จ] บันทึกกราฟประเมินผลไว้ที่: {save_path}")

if __name__ == "__main__":
    evaluate_multi_task_model()