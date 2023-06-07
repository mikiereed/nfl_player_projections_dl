import os.path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from utils import split_dataset, get_columns


def linear_regression_model_multi_mse(data_path: str, normalize_data: bool) -> float:
    x_train, x_val, x_test, y_train, y_val, y_test, y_scaler = split_dataset(data_path=data_path,
                                                                             normalize_data=normalize_data,
                                                                             )
    feature_columns, target_columns, _, _ = get_columns(data_path=data_path)

    y_pred = np.zeros(y_test.shape)
    for idx, target in enumerate(target_columns):
        y_i_train = y_train[:, idx]
        lr = LinearRegression()
        lr.fit(x_train, y_i_train)
        y_i_pred = lr.predict(x_test)
        y_pred[:, idx] = y_i_pred

    # undo normalization
    if normalize_data:
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)

    test_mse = mean_squared_error(y_pred=y_pred, y_true=y_test)

    return test_mse


if __name__ == "__main__":
    print("previous years,normalize,unnormalized")
    for i in range(1, 6):
        norm_mse = linear_regression_model_multi_mse(
            data_path=os.path.join("data", f"clean_data_{i}.csv"),
            normalize_data=True,
        )
        unnorm_mse = linear_regression_model_multi_mse(
            data_path=os.path.join("data", f"clean_data_{i}.csv"),
            normalize_data=False,
        )
        print(f"{i},{unnorm_mse},{norm_mse}")
