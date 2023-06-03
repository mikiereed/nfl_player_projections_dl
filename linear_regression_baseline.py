import os.path

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from utils import load_dataset


STATS_TO_BASELINE = [
        "Passing Yards",
        "TD Passes",
        "Ints",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
        "Fumbles",
    ]


def linear_regression_model(data_path: str, trials: int, random_seed: int, normalize_data: bool = False):
    for stat in STATS_TO_BASELINE:
        stat = f"Target {stat}"
        x, y, _, _ = load_dataset(csv_path=data_path, feature_columns=None, target_columns=[stat], add_intercept=False)
        if normalize_data:
            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            x = x_scaler.fit_transform(x)
            y = y_scaler.fit_transform(np.reshape(y, (-1, 1)))

        reg_scores = []
        mes = []
        for i in range(random_seed, trials + random_seed):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)

            lr = LinearRegression()
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            reg_score = lr.score(x_test, y_test)
            reg_scores.append(reg_score)
            mes.append(metrics.mean_squared_error(y_test, y_pred))

        print(f"Baseline LR {stat}")
        # print(f"reg scores: {sum(reg_scores) / trials}")
        print(f"{sum(mes) / trials}")


if __name__ == "__main__":
    linear_regression_model(
        # data_x="C:\\Users\\mikie\\OneDrive\\stanford homework\\cs230\\final project\\clean_data.csv",
        data_path=os.path.join("data", "clean_data.csv"),
        trials=100,
        random_seed=88,
        normalize_data=False,
    )
