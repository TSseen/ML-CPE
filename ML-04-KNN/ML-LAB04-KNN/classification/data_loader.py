


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

    X = df[NUMERIC_FEATURES].copy()
   # convert result (target = quality score) to class index
    class_names = sorted(df["label"].unique())
    y = df["label"].map({name: i for i, name in enumerate(class_names)})

    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    # step 3 : split data เป็น train 60 / validation 20 / test 20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp)

    # step 4 : Scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": NUMERIC_FEATURES ,
        "n_rows": len(df),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("คลาส  :", data["class_names"])