import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.tab_model import TabNetRegressor

from utils import load_dataset


def tabnet(data_path: str, normalize_data: bool = False):
    clf = TabNetRegressor()

    X, Y, feature_cols, target_cols = load_dataset(data_path)
    if normalize_data:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X = x_scaler.fit_transform(X)
        Y = y_scaler.fit_transform(Y)

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=88)

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=88)

    max_epochs = 1000 if not os.getenv("CI", False) else 2

    clf.fit(
        X_train=x_train, y_train=y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['mse'],
        # eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        max_epochs=max_epochs,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    saving_path_name = os.path.join("models", "tabnet_model_test_2")
    saved_filepath = clf.save_model(saving_path_name)
    print(f"saved at {saved_filepath}")

    preds = clf.predict(x_test)

    test_mse = mean_squared_error(y_pred=preds, y_true=y_test)

    print(f"BEST VAL SCORE: {clf.best_cost}")
    print(f"FINAL TEST SCORE: {test_mse}")

    clf.feature_importances_

    explain_matrix, masks = clf.explain(x_test)

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")

    plt.show()


if __name__ == '__main__':
    _data_path = os.path.join("data", "clean_data.csv")
    tabnet(_data_path, normalize_data=True)
