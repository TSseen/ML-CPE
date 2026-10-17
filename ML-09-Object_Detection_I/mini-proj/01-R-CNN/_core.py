from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 64
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
COLORS = [(0, 200, 0), (0, 0, 255), (0, 200, 255)]  # BGR
MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "model.keras"


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def detect_demo(model, frame, threshold=0.85, regions=25):
    h, w = frame.shape[:2]
    sizes = (0.3, 0.5)
    per_size = max(1, regions // len(sizes))
    g = max(1, int(np.ceil(np.sqrt(per_size))))

    boxes, crops = [], []
    for s in sizes:
        bs = int(min(h, w) * s)
        for y in np.linspace(0, h - bs, g).astype(int):
            for x in np.linspace(0, w - bs, g).astype(int):
                crop = frame[y:y + bs, x:x + bs]
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop)
                boxes.append((x, y, x + bs, y + bs))

    batch = np.asarray(crops, dtype=np.float32) / 255.0
    probs = model.predict(batch, verbose=0)

    preds = []
    for box, p in zip(boxes, probs):
        cls = int(np.argmax(p))
        conf = float(p[cls])
        if conf >= threshold:
            preds.append((*box, cls, conf))
    return preds


def draw(frame, preds):
    out = frame.copy()
    for x1, y1, x2, y2, cls, conf in preds:
        color = COLORS[cls]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{CLASSES[cls]} {conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out