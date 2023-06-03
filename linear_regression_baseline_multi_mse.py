import os.path

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from utils import load_dataset


def linear_regression_model_multi_mse(data_path: str, trials: int, random_seed: int, normalize_data: bool = False):
    x, y, feature_columns, target_columns = load_dataset(csv_path=data_path)
    if normalize_data:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        x = x_scaler.fit_transform(x)
        y = y_scaler.fit_transform(y)

    test_mse = 0
    for i in range(random_seed, trials + random_seed):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=i)
        y_pred = np.zeros(y_test.shape)
        for idx, target in enumerate(target_columns):
            y_i_train = y_train[:, idx]
            lr = LinearRegression()
            lr.fit(x_train, y_i_train)
            y_i_pred = lr.predict(x_test)
            y_pred[:, idx] = y_i_pred

        test_mse += mean_squared_error(y_pred=y_pred, y_true=y_test)
        if i % 20 == 0:
            print(i)

    # print(f"Baseline LR {stat}")
    print(f"{test_mse / trials}")


if __name__ == "__main__":
    linear_regression_model_multi_mse(
        # data_x="C:\\Users\\mikie\\OneDrive\\stanford homework\\cs230\\final project\\clean_data.csv",
        data_path=os.path.join("data", "clean_data.csv"),
        trials=100,
        random_seed=88,
        normalize_data=True,
    )
