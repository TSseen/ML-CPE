import os, random, shutil
import xml.etree.ElementTree as ET

OUT = "dataset"
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

files = [f[:-4] for f in os.listdir("annotations") if f.endswith(".xml")]
random.seed(42)
random.shuffle(files)
n = len(files)
splits = {
    "train": files[:int(n * 0.7)],
    "val": files[int(n * 0.7):int(n * 0.85)],
    "test": files[int(n * 0.85):],
}

for split, names in splits.items():
    os.makedirs(f"{OUT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT}/{split}/labels", exist_ok=True)
    for name in names:
        root = ET.parse(f"annotations/{name}.xml").getroot()
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)
        lines = []
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASSES:
                continue
            b = obj.find("bndbox")
            x1, y1, x2, y2 = [float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{CLASSES.index(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        shutil.copy(f"images/{name}.png", f"{OUT}/{split}/images/{name}.png")
        with open(f"{OUT}/{split}/labels/{name}.txt", "w") as f:
            f.write("\n".join(lines))

with open(f"{OUT}/data.yaml", "w") as f:
    f.write("path: dataset\ntrain: train/images\nval: val/images\ntest: test/images\n")
    f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")