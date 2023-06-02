# major code ideas from https://github.com/dreamquark-ai/tabnet/blob/develop/pretraining_example.ipynb

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.pretraining import TabNetPretrainer

from utils import load_dataset


def tabnet_pretrain(X: np.ndarray, Y: np.ndarray, normalize_data: bool = False) -> None:
    unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',  # "sparsemax",
        n_shared_decoder=1,  # nb shared glu for decoding
        n_indep_decoder=1,  # nb independent glu for decoding
        #     grouped_features=[[0, 1]], # you can group features together here
        verbose=5,
    )

    if normalize_data:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X = x_scaler.fit_transform(X)
        Y = y_scaler.fit_transform(Y)

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=88)

    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=88)

    unsupervised_model.fit(
        X_train=x_train,
        eval_set=[x_val],
        max_epochs=1000,
        patience=5,
        batch_size=2048,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.5,
    )

    save_path_name = os.path.join("models", "pretrained")
    unsupervised_model.save_model(save_path_name)


if __name__ == '__main__':
    data_path = os.path.join("data", "clean_data.csv")
    _X, _Y, feature_cols, target_cols = load_dataset(data_path)
    tabnet_pretrain(_X, _Y, normalize_data=False)  # run multitask model

