
"""
Read CSV
convert text to number
make Scaling for KNN
split data: train / validation / test
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "WineQT.csv"    

TARGET = "quality"

# config feature is numeric features that are already numbers
NUMERIC_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates", 
    "alcohol"
]


def quality_to_label(q):
    if q <= 5:
        return "low"
    elif q == 6:
        return "medium"
    else:
        return "high"

# ---------------------------------------------------------------------------
def load_data(test_size=0.2, seed=42):
    
    # step 1 : read CSV
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()            
    df["label"] = df[TARGET].apply(quality_to_label)

    X_raw = df[NUMERIC_FEATURES].copy().to_numpy(dtype="float32")
    
    # step 4 : Scaling สำหรับ Clustering และการใช้งานทั่วไป
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype("float32")

    # แปลง label เผื่อไว้ใช้ฝั่ง Classification
    class_names = sorted(df["label"].unique())
    y = df["label"].map({name: i for i, name in enumerate(class_names)}).to_numpy(dtype="int32")

    # ส่งคืนค่าให้รองรับทั้ง Clustering (ใช้ X, X_raw) และ Classification (ใช้ X_train, etc.)
    return {
        "X": X_scaled,
        "X_scaled": X_scaled,
        "X_raw": X_raw,
        "dataframe": df,
        "df": df,
        "feature_names": NUMERIC_FEATURES,
        "features": NUMERIC_FEATURES,
        "n_rows": len(df),
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ปรับเรียกใช้ฟังก์ชันโหลดแบบที่เหมาะสมกับ Clustering หรือเขียนฟังก์ชันแปลงแยก
    df = pd.read_csv(CSV_PATH).dropna()
    X_raw = df[NUMERIC_FEATURES].to_numpy(dtype="float32")
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype("float32")

    print("size data :", X.shape)
    print("mean after scale (should be close to 0) :", X.mean(axis=0).round(3))