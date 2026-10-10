import os
import splitfolders

def split_dataset(input_folder, output_folder, ratio=(0.8, 0.1, 0.1)):
    print("กำลังแบ่งข้อมูล...")
    # แบ่งข้อมูลเป็น Train, Val, Test ตามสัดส่วน
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=ratio, group_prefix=None)
    print(f"แบ่งข้อมูลสำเร็จ! บันทึกไว้ที่: {output_folder}")

if __name__ == "__main__":
    # ใส่ Path โฟลเดอร์รูปภาพต้นฉบับ

    TRAIN_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\train"
    VAL_DIR = r"C:\Users\tiraw\OneDrive\เดสก์ท็อป\ML\ML-08-DCNN\mini-proj\New Plant Diseases Dataset(Augmented)\valid"
    SPLIT_DATA_PATH = "dataset_split"
    
    if os.path.exists(RAW_DATA_PATH):
        split_dataset(RAW_DATA_PATH, SPLIT_DATA_PATH)
    else:
        print("โปรดระบุที่อยู่ของโฟลเดอร์รูปภาพในตัวแปร RAW_DATA_PATH")