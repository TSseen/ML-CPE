#STEP 1  Load the dataset.
#STEP 2  Find the best k using the Elbow Method.
#STEP 3  Run K-Means with the selected k.
#STEP 4  Analyze each cluster.
#STEP 5  Use KNN to classify a new animal.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import KNNClusterAssigner

OUT_DIR = Path(__file__).resolve().parent / "outputs"

N_CLUSTERS = 4      
KNN_K = 5           


def title(text):
    print("\n" + "--" * 30)
    print(text)


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    title("STEP 1 : load data")
    data = data_loader.load_data()
    X = data["X"]           # ใช้ข้อมูลหลัง scale สำหรับคำนวณโมเดล
    X_raw = data["X_raw"]          # ข้อมูลหน่วยจริงสำหรับใช้อธิบายผล
    df = data["dataframe"]
    features = data["feature_names"]

    print(f"size data : {X.shape[0]} แถว x {X.shape[1]} feature")
    print("feature is clustering :")
    for f in features:
        print(f"    - {f}")

    # =====================================================================
    title("STEP 2 : should we split into how many clusters? ")
    # =====================================================================
    k_values = [2, 3, 4, 5, 6, 7, 8]
    inertias = []

    for k in k_values:
        km = TFKMeans(n_clusters=k)
        labels_temp = km.fit_predict(X)
        sil = silhouette_score(X, labels_temp)
        inertias.append(km.inertia_)
        print(f"   k = {k}  ->  inertia = {km.inertia_:8.1f}    silhouette = {sil:.3f}")

    visualize.plot_elbow(k_values, inertias, OUT_DIR / "01_elbow.png")
    print(f"\n plot graph outputs/01_elbow.png and select k is print in graph ")
    print(f" select k = {N_CLUSTERS} in the graph")

    # =====================================================================
    title(f"STEP 3 : Run K-Means (k = {N_CLUSTERS})")
    # =====================================================================
    km = TFKMeans(n_clusters=N_CLUSTERS)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"Inertia          : {km.inertia_:.1f}")
    print(f"Silhouette score : {sil:.3f}")
    print(f"member number in each cluster : {np.bincount(labels).tolist()}")

    if sil < 0.25:
            print("\n[Note] A low silhouette score means weak clusters.")
            print("       This dataset has no clear natural groups.")
            print("       K-Means always creates clusters.")
            print("       Always check the silhouette score.")

    # พล็อตเลือกฟีเจอร์คอลัมน์ที่ 0 และ 1 มาแสดงผลในกราฟกระจายตัว
    visualize.plot_clusters(X_raw[:, [0, 1]], labels, OUT_DIR / "02_clusters.png",
                            x_name=features[0], y_name=features[1])

    title("STEP 4 : What are the characteristics of each group??")

    profile = pd.DataFrame(X_raw.astype("float64"), columns=features)
    profile["cluster"] = labels

    summary = profile.groupby("cluster").mean().round(1)
    summary["member count"] = np.bincount(labels)

    print(summary.to_string())
    summary.to_csv(OUT_DIR / "cluster_summary.csv", encoding="utf-8-sig")

    title(f"STEP 5 : use KNN detect animal into groups (k = {KNN_K})")
  
    # จำลองสถานการณ์: แบ่งข้อมูลส่วนหนึ่งเป็นตัวอย่าง และส่วนหนึ่งเป็นข้อมูลใหม่
    n_known = int(len(X) * 0.8)
    X_known, labels_known = X[:n_known], labels[:n_known]
    X_new, labels_new = X[n_known:], labels[n_known:]

    assigner = KNNClusterAssigner(k=KNN_K)
    assigner.fit(X_known, labels_known)
    knn_pred = assigner.predict(X_new)

    accuracy = float(np.mean(knn_pred == labels_new))
    print(f"number of 'new data' : {len(X_new)} แถว")
    print(f"KNN detect groups correctly compared to K-Means : {accuracy * 100:.1f} %")
    print("Well-separated clusters give better KNN classification.")
    print("Use KNN for new data without rerunning K-Means.")

    # =====================================================================
    title("save results to CSV file")
    # =====================================================================
    result = df.copy()
    result["cluster"] = labels        # เพิ่มคอลัมน์บอกกลุ่มคลัสเตอร์
    result.to_csv(OUT_DIR / "clustered_animals.csv",
                  index=False, encoding="utf-8-sig")

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()