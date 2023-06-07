# major code ideas from https://github.com/dreamquark-ai/tabnet/blob/develop/multi_regression_example.ipynb

import os
from typing import Union

from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer

from utils import split_dataset, get_columns


def tabnet(data_path: str, normalize_data: bool, pretrained_model_path: Union[str, None], save_path: str) -> float:
    """
    Creates a tabnet model based on given parameters and data_set
    :param data_path: data_set path
    :param normalize_data: should data be normalized
    :param pretrained_model_path: path of pretrained model. None means don't use a pretrained model
    :param save_path: path to save model
    :return: mse for model
    """
    clf = TabNetRegressor()

    x_train, x_val, x_test, y_train, y_val, y_test, y_scaler = split_dataset(data_path=data_path,
                                                                             normalize_data=normalize_data,
                                                                             )

    if pretrained_model_path:
        loaded_pretrain = TabNetPretrainer()
        loaded_pretrain.load_model(pretrained_model_path)
    else:
        loaded_pretrain = None

    clf.fit(
        X_train=x_train, y_train=y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_name=['train', 'valid'],
        # eval_metric=['rmsle'],
        eval_metric=['mse'],
        max_epochs=1000,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        from_unsupervised=loaded_pretrain,
    )

    clf.save_model(save_path)
    print(f"Saved at: {save_path}")

    predictions = clf.predict(x_test)
    test_mse = mean_squared_error(y_pred=predictions, y_true=y_test)
    print(f"Model MSE: {test_mse}")

    return test_mse


if __name__ == '__main__':
    for i in range(1, 6):
        _data_path = os.path.join("data", f"clean_data_{i}.csv")
        for pretrained in [True, False]:
            pretrained_value = "pretrained_" if pretrained else ""
            for normalized in [True, False]:
                normalized_value = "normalized" if normalized else "unnormalized"
                pretrained_model = None if not pretrained else os.path.join("models", f"pretrained_{normalized_value}_{i}.zip")
                _save_path = os.path.join("models", f"tabnet_{pretrained_value}{normalized_value}_{i}")
                tabnet(data_path=_data_path,
                       normalize_data=normalized,
                       pretrained_model_path=pretrained_model,
                       save_path=_save_path
                       )

    # # run tabnet for each individual Y
    # for normalized in ["normalized", "unnormalized"]:
    #     normalize_data = (normalized == "normalized")
    #     for target in target_cols:
    #         print(f"{normalized} {normalize_data} {target}")
    #         _X, _Y, feature_cols, target_cols = load_dataset(data_path, target_columns=[target])
    #         mse = tabnet(_X, _Y.reshape(-1, 1), normalize_data=normalize_data, from_unsupervised=True)
    #         with open(results_file, "a") as f:
    #             f.write(f"{target} (pretrained, {normalized})\n")
    #             f.write(f"{mse}\n")
