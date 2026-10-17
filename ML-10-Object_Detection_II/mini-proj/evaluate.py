from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # 1. ชี้ไปที่โมเดล mask_detect ที่คุณก๊อปปี้มาไว้ด้านนอก (และมีระบบสำรองเช็กใน runs ให้ด้วย)
    weights = ROOT / 'mask_detect' / 'weights' / 'best.pt'
    if not weights.exists():
        weights = ROOT / 'runs' / 'detect' / 'mask_detect' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))

    # 2. ประเมินผลกับชุดข้อมูล test (ชี้ไปที่ data.yaml ด้านนอกสุด และใช้ imgsz=640 ตามตอนเทรน)
    metrics = model.val(
        data=str(ROOT / 'data.yaml'),
        split='test',
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / 'runs'),
        name='evaluation',
        exist_ok=True,
    )

    # 3. แสดงผลลัพธ์ความแม่นยำ
    print('\n===== Evaluation Results =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')


if __name__ == '__main__':
    main()