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
        loss={
            'gender_out': 'binary_crossentropy', 
            'age_out': 'mse'
        },
        loss_weights={
            'gender_out': 6.0,  # บังคับให้ AI ให้ความสำคัญกับเพศมากขึ้น 5 เท่า
            'age_out': 0.2     # ลดความสำคัญของอายุลง เพื่อไม่ให้ดึง Error ไปหมด
        },
        metrics={
            'gender_out': 'accuracy', 
            'age_out': 'mae'
        }
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
    # --- ส่วนที่เพิ่มใหม่: สร้างกราฟ Training History สำหรับโมเดล 2 หัว (Multi-task) ---
    # --- ฟังก์ชันสำหรับทำ Smoothing กราฟให้เรียบเนียน ---
    def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    # --- เริ่มวาดกราฟแบบ Multi-task ---
    print("\n6. กำลังสร้างกราฟ Training History แบบสมูท...")
    
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Multi-task CNN Training Performance (Smoothed)', fontsize=18, fontweight='bold', y=1.02)

    # ดึงข้อมูลจาก history
    epochs = range(len(history.history['gender_out_accuracy']))
    
    acc = history.history['gender_out_accuracy']
    val_acc = history.history['val_gender_out_accuracy']
    loss = history.history['gender_out_loss']
    val_loss = history.history['val_gender_out_loss']
    mae = history.history['age_out_mae']
    val_mae = history.history['val_age_out_mae']
    t_loss = history.history['loss']
    val_t_loss = history.history['val_loss']

    # 1. กราฟ Gender Accuracy
    axes[0, 0].plot(epochs, acc, alpha=0.3, color='dodgerblue') # เส้นดิบจางๆ
    axes[0, 0].plot(epochs, smooth_curve(acc), label='Train Gender Acc (Smooth)', color='dodgerblue', linewidth=2.5)
    axes[0, 0].plot(epochs, val_acc, alpha=0.3, color='darkorange')
    axes[0, 0].plot(epochs, smooth_curve(val_acc), label='Val Gender Acc (Smooth)', color='darkorange', linewidth=2.5)
    axes[0, 0].set_title('Gender Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].legend()

    # 2. กราฟ Gender Loss
    axes[0, 1].plot(epochs, loss, alpha=0.3, color='dodgerblue')
    axes[0, 1].plot(epochs, smooth_curve(loss), label='Train Gender Loss (Smooth)', color='dodgerblue', linewidth=2.5)
    axes[0, 1].plot(epochs, val_loss, alpha=0.3, color='darkorange')
    axes[0, 1].plot(epochs, smooth_curve(val_loss), label='Val Gender Loss (Smooth)', color='darkorange', linewidth=2.5)
    axes[0, 1].set_title('Gender Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend()

    # 3. กราฟ Age MAE
    axes[1, 0].plot(epochs, mae, alpha=0.3, color='mediumseagreen')
    axes[1, 0].plot(epochs, smooth_curve(mae), label='Train Age MAE (Smooth)', color='mediumseagreen', linewidth=2.5)
    axes[1, 0].plot(epochs, val_mae, alpha=0.3, color='crimson')
    axes[1, 0].plot(epochs, smooth_curve(val_mae), label='Val Age MAE (Smooth)', color='crimson', linewidth=2.5)
    axes[1, 0].set_title('Age Regression Error (MAE in Years)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()

    # 4. กราฟ Total Loss
    axes[1, 1].plot(epochs, t_loss, alpha=0.3, color='purple')
    axes[1, 1].plot(epochs, smooth_curve(t_loss), label='Train Total Loss (Smooth)', color='purple', linewidth=2.5)
    axes[1, 1].plot(epochs, val_t_loss, alpha=0.3, color='violet')
    axes[1, 1].plot(epochs, smooth_curve(val_t_loss), label='Val Total Loss (Smooth)', color='violet', linewidth=2.5)
    axes[1, 1].set_title('Total Model Loss', fontsize=14, fontweight='bold')
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

    plt.tight_layout()
    plt.savefig("outputs/training_history_smoothed.png", dpi=300, bbox_inches='tight')
    print("[สำเร็จ] บันทึกกราฟสมูทแล้วไว้ที่: outputs/training_history_smoothed.png")
    # -------------------------------------------------------------------
if __name__ == "__main__":
    main()