import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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


def linear_regression_model(data_x: str, data_y, trials: int, random_seed: int):
    for stat in STATS_TO_BASELINE:
        x, y = load_dataset(csv_path_x=data_x, csv_path_y=data_y, y_column=stat, add_intercept=True)

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

        print(f"stat: {stat}")
        print(f"reg scores: {sum(reg_scores) / trials}")
        print(f"mes: {sum(mes) / trials}")


def add_intercept_fn(x):
    x_with_intercept = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    x_with_intercept[:, 0] = 1
    x_with_intercept[:, 1:] = x

    return x_with_intercept


def load_dataset(csv_path_x, csv_path_y, y_column='y', add_intercept=False):
    with open(csv_path_x, 'r', newline='') as f:
        x_cols = f.readline().strip().split(',')
    x_cols = [i for i in range(len(x_cols))]

    with open(csv_path_y, 'r', newline='') as f:
        y_cols = f.readline().strip().split(',')
    l_cols = list()
    l_cols.append(y_cols.index(y_column))

    inputs = np.loadtxt(csv_path_x, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path_y, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


if __name__ == "__main__":
    linear_regression_model(
        data_x="C:\\Users\\mikie\\OneDrive\\stanford homework\\cs230\\final project\\clean_data_X.csv",
        data_y="C:\\Users\\mikie\\OneDrive\\stanford homework\\cs230\\final project\\clean_data_Y.csv",
        trials=10,
        random_seed=88,
    )
