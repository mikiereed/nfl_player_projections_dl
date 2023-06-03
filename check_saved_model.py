import copy
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer

from utils import load_dataset


def check_saved_model(saved_model: str, x_test: np.ndarray, y_test: np.ndarray) -> float:
    clf = TabNetRegressor()
    clf.load_model(saved_model)

    preds = clf.predict(x_test)

    # # test getting sum of each mse
    # test_mse = 0
    # for i in range(preds.shape[1]):
    #     test_mse += mean_squared_error(y_pred=preds[:, i], y_true=y_test[:, i])

    test_mse = mean_squared_error(y_pred=preds, y_true=y_test)

    print(f"FINAL TEST SCORE: {test_mse}")

    return test_mse


def _get_test_data(data_path: str, normalize_data=True) -> tuple:
    X, Y, feature_cols, target_cols = load_dataset(data_path)

    if normalize_data:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X = x_scaler.fit_transform(X)
        Y = y_scaler.fit_transform(Y)
    else:
        y_scaler = None

    # make sure random state is the same as model creation random state in tabnet.py
    # otherwise, test group will include data from training/validation sets
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=88)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=88)

    return x_test, y_test, y_scaler


if __name__ == '__main__':
    mse = dict()
    for i in range(1, 6):  # change to match years to check
        data_path = os.path.join("data", f"clean_data_{i}.csv")
        x_test_normalized, y_test_normalized, y_scaler = _get_test_data(data_path, normalize_data=True)
        x_test_unnormalized, y_test_unnormalized, _ = _get_test_data(data_path, normalize_data=True)

        for model in os.listdir("models"):
            if not model.startswith("tabnet"):
                continue  # this is just pretraining weights
            model_path = os.path.join("models", model)
            model_info = model.split("_")
            normalized_value = model_info[-2]
            year = int(model_info[-1].split(".")[0])
            if year != i:
                continue  # using the wrong dataset
            if len(model_info) == 4:  # has pretrained in the name
                pretrained_value = "pretrained"
            else:
                pretrained_value = "not pretrained"

            if normalized_value == "normalized":
                x_test = np.copy(x_test_normalized)
                y_test = np.copy(y_test_normalized)
            else:
                x_test = np.copy(x_test_unnormalized)
                y_test = np.copy(y_test_unnormalized)
            model_mse = check_saved_model(model_path, x_test, y_test)
            if mse.get(year):
                mse[year][f"{pretrained_value}, {normalized_value}"] = model_mse
            else:
                mse[year] = {f"{pretrained_value}, {normalized_value}": model_mse}

            # results_file = os.path.join(".", "results.txt")
            # with open(results_file, "a") as f:
            #     f.write(f"Tabnet all (normalized, pretrained, summed)\n")
            #     f.write(f"{mse}\n")
