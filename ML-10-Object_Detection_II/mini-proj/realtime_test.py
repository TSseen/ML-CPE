from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # 1. ชี้ไปที่โมเดล mask_detect ที่คุณเทรนเสร็จแล้ว (และมีระบบสำรองเช็กใน runs ให้ด้วย)
    weights = ROOT / 'mask_detect' / 'weights' / 'best.pt'
    if not weights.exists():
        weights = ROOT / 'runs' / 'detect' / 'mask_detect' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    # ตรวจสอบการใช้งานการ์ดจออัตโนมัติ
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # โหลดโมเดล
    model = YOLO(str(weights))

    # เปิดกล้อง Webcam (กล้องหลัก index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam (camera index 0)')

    print('\nStarting real-time detection... Press "q" to exit.')

    try:
        while True:
            # อ่านภาพจากกล้องทีละเฟรม
            ok, frame = cap.read()

            if not ok:
                print('Failed to grab frame')
                break

            # ตรวจจับวัตถุด้วย YOLO
            results = model.predict(
                source=frame,
                conf=0.50,          # ปรับความมั่นใจขั้นต่ำ (Confidence) ได้ที่นี่ (เช่น 0.50 หรือ 0.25)
                imgsz=640,
                device=device,
                verbose=False
            )

            # วาดกรอบสี่เหลี่ยมลงบนเฟรม
            annotated_frame = results[0].plot()

            # แสดงผลลัพธ์ผ่านหน้าต่าง OpenCV
            cv2.imshow(
                'Face Mask Real-time Detection',
                annotated_frame
            )

            # กดปุ่ม 'q' หรือปุ่ม 'Esc' (รหัส 27) เพื่อปิดกล้องและออกจากโปรแกรม
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break

    finally:
        # ปิดกล้องและเคลียร์หน้าต่างทั้งหมดเมื่อเลิกใช้งาน
        cap.release()
        cv2.destroyAllWindows()
        print('\nWebcam testing completed!')


if __name__ == '__main__':
    main()