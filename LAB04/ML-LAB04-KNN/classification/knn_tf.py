import numpy as np
import tensorflow as tf


class TFKNNClassifier:

    def __init__(self, k=5):
        self.k = k
#สร้าง class ชื่อ TFKNNClassifier มีค่าตั้งต้นคือ k (จำนวนเพื่อนบ้านที่จะดู) ตอนสร้าง object ถ้าไม่ระบุจะใช้ k=5
    def fit(self, X, y):
        # เก็บ training data ไว้เฉยๆ ไม่มีการเทรนจริง
        self.X_train = tf.constant(X, dtype=tf.float32)
        self.y_train = tf.constant(y, dtype=tf.int32)
        self.n_classes = int(y.max()) + 1
        return self

    def _distance(self, X_new): #คำนวณระยะห่าง
        # คำนวณระยะห่างจากทุกจุดใหม่ไปยังทุกจุดใน training set
        diff = X_new[:, None, :] - self.X_train[None, :, :]
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))
    #สูตร sqrt((x1-y1)² + (x2-y2)² + ...)
    
    def predict(self, X):
        X = tf.constant(X, dtype=tf.float32)
        dist = self._distance(X)

        # หา k เพื่อนบ้านที่ใกล้ที่สุด (top_k ใช้ -dist เพราะ top_k หาค่ามากสุด แต่เราต้องการค่าน้อยสุด)
        _, idx = tf.math.top_k(-dist, k=self.k)
        neighbor_labels = tf.gather(self.y_train, idx)

        # โหวตเสียงข้างมากจาก k เพื่อนบ้าน
        onehot = tf.one_hot(neighbor_labels, depth=self.n_classes)
        votes = tf.reduce_sum(onehot, axis=1)

        return tf.argmax(votes, axis=1).numpy()

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()
    model = TFKNNClassifier(k=5)
    model.fit(data["X_train"], data["y_train"]) #"เทรน" โมเดล

    acc = model.score(data["X_val"], data["y_val"])
    print(f"Validation accuracy (k=5): {acc:.4f}")