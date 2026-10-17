from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # 1. ชี้ไปที่โมเดล mask_detect ที่คุณก๊อปปี้มาไว้ด้านนอก (และสำรองเช็กใน runs ให้ด้วย)
    weights = ROOT / 'mask_detect' / 'weights' / 'best.pt'
    if not weights.exists():
        weights = ROOT / 'runs' / 'detect' / 'mask_detect' / 'weights' / 'best.pt'

    # 2. โฟลเดอร์รูปภาพสำหรับทดสอบ
    image_dir = ROOT / 'dataset' / 'test' / 'images'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    if not image_dir.exists():
        raise FileNotFoundError(f'Dataset not found: {image_dir}')

    # โหลดโมเดล
    model = YOLO(str(weights))

    # ดึงรายชื่อไฟล์รูปภาพทั้งหมด (.jpg, .jpeg, .png)
    extensions = {'.jpg', '.jpeg', '.png'}
    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ]

    if not images:
        raise RuntimeError('No test images found')

    # สุ่มเลือกรูปภาพมา 5 รูป
    selected_images = random.sample(
        images, min(5, len(images))
    )

    # เริ่มตรวจจับและบันทึกผลลัพธ์ลงใน runs/image_test
    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=640,  # ปรับเป็น 640 ให้ตรงกับขนาดภาพตอนเทรน
            device=0,
            save=True,
            project=str(ROOT / 'runs'),
            name='image_test',
            exist_ok=True
        )

        for result in results:
            print(f'\nImage: {image.name}')

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                print(
                    f'{model.names[class_id]}: '
                    f'{confidence:.2f}'
                )

            # แสดงหน้าต่างรูปภาพที่ตีกรอบแล้ว
            annotated = result.plot()
            cv2.imshow('YOLO Image Detection', annotated)

            # กดปุ่มอะไรก็ได้บนคีย์บอร์ดเพื่อดูรูปถัดไป (หรือกด 'q' เพื่อออก)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nImage testing completed!')


if __name__ == '__main__':
    main()