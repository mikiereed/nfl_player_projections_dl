import csv
import os
from typing import Optional

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.tab_model import TabNetRegressor

from utils import split_dataset


def check_saved_model(
        saved_model: str,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_scaler: Optional[MinMaxScaler],
) -> float:
    """
    Load a saved model .zip and check results against test set.
    Use utils.split_dataset for x_test, and y_test to ensure we get a test set that was not used in training
    """
    clf = TabNetRegressor()
    clf.load_model(saved_model)

    predictions = clf.predict(x_test)

    # de-normalization
    if y_scaler:
        predictions = y_scaler.inverse_transform(predictions)
        y_test = y_scaler.inverse_transform(y_test)

    test_mse = mean_squared_error(y_pred=predictions, y_true=y_test)

    return test_mse


if __name__ == '__main__':
    seed = 93
    mse = dict()
    headers = ["previous years"]
    for i in range(1, 6):  # change to match years to check
        data_path = os.path.join("data", f"clean_data_{i}.csv")
        _, _, x_test_normalized, _, _, y_test_normalized, y_scaler = split_dataset(data_path, normalize_data=True)
        _, _, x_test_unnormalized, _, _, y_test_unnormalized, _ = split_dataset(data_path, normalize_data=False)

        for model in os.listdir(f"models_{seed}"):
            if not model.startswith("tabnet"):
                continue  # this is just pretraining weights
            model_path = os.path.join(f"models_{seed}", model)
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
                _x_test = np.copy(x_test_normalized)
                _y_test = np.copy(y_test_normalized)
                _y_scaler = y_scaler
            else:
                _x_test = np.copy(x_test_unnormalized)
                _y_test = np.copy(y_test_unnormalized)
                _y_scaler = None

            model_mse = check_saved_model(model_path, _x_test, _y_test, _y_scaler)

            label = f"{pretrained_value}, {normalized_value}"
            if label not in headers:
                headers.append(label)
            if mse.get(year):
                mse[year][label] = model_mse
            else:
                mse[year] = {label: model_mse}

    # save results to a csv
    results_file = os.path.join(".", f"results_{seed}_denormed.csv")
    with open(results_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        for year, models in mse.items():
            row = [year]
            for i in range(1, len(headers)):
                row.append(models[headers[i]])
            csvwriter.writerow(row)
