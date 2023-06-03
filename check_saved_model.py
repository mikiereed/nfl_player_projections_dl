import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer

from utils import load_dataset


def check_saved_model(saved_model: str, X: np.ndarray, Y: np.ndarray, normalize_data: bool) -> float:
    clf = TabNetRegressor()
    clf.load_model(saved_model)

    if normalize_data:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X = x_scaler.fit_transform(X)
        Y = y_scaler.fit_transform(Y)

    # keep random state the same as model creation to ensure only using test set
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=88)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=88)

    preds = clf.predict(x_test)

    test_mse = 0

    for i in range(preds.shape[1]):
        test_mse += mean_squared_error(y_pred=preds[:, i], y_true=y_test[:, i])

    print(f"FINAL TEST SCORE: {test_mse}")

    return test_mse


if __name__ == '__main__':
    data_path = os.path.join("data", "clean_data.csv")
    _X, _Y, feature_cols, target_cols = load_dataset(data_path)
    mse = check_saved_model(os.path.join(".", "models", "tabnet_pretrained_normalized.zip"), _X, _Y, normalize_data=True)  # run multitask model
    results_file = os.path.join(".", "results.txt")
    with open(results_file, "a") as f:
        f.write(f"Tabnet all (normalized, pretrained, summed)\n")
        f.write(f"{mse}\n")
