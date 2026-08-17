import os
import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# เลือก 5 ตัวละครที่มีภาพเยอะสุดใน dataset เพื่อให้ class สมดุลกัน
SELECTED_CHARACTERS = [
    "homer_simpson",
    "bart_simpson",
    "lisa_simpson",
    "marge_simpson",
    "krusty_the_clown",
]


def load_data(data_path, img_size=64, max_per_class=300,
              characters=SELECTED_CHARACTERS):

    images = []
    labels = []

    classes = characters
    print("Selected classes:", classes)

    for label, character in enumerate(classes):
        class_path = os.path.join(data_path, character)
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            try:
                image_array = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception:
                image = None

            image = preprocess_image(image, img_size)

            # Skip unreadable or damaged images
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {character}: {loaded} images ({skipped} skipped)")

    return np.stack(images), np.array(labels), classes
