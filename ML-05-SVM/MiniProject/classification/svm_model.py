from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

KERNELS = ["linear", "poly", "rbf"]


def train_svm_all_kernels(X_train, y_train):
    """Standardize once, then train one SVM per kernel.

    Returns (models, scaler) where models is {kernel_name: fitted_model}.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}
    for kernel in KERNELS:
        print(f"Training SVM with kernel = {kernel} ...")
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train_scaled, y_train)
        models[kernel] = model

    return models, scaler


def predict_svm(model, scaler, X_test):
    # Apply the same scaling used for training data
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return predictions
