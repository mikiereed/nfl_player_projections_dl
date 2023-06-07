# major code ideas from https://github.com/dreamquark-ai/tabnet/blob/develop/pretraining_example.ipynb

import os
import torch
import numpy as np
from pytorch_tabnet.pretraining import TabNetPretrainer

from utils import split_dataset


def tabnet_pretrain(data_path: str, normalize_data: bool, save_path: str) -> None:
    """
    Pretrain a model using tabnet pretrainer
    :param data_path: path to data csv
    :param normalize_data: should data be normalized?
    :param save_path: path to save model
    :return: None
    """
    unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',  # "sparsemax",
        n_shared_decoder=1,  # nb shared glu for decoding
        n_indep_decoder=1,  # nb independent glu for decoding
        verbose=5,
    )

    x_train, x_val, *_ = split_dataset(data_path=data_path, normalize_data=normalize_data)

    unsupervised_model.fit(
        X_train=x_train,
        eval_set=[x_val],
        max_epochs=1000,
        patience=10,
        batch_size=2048,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.5,
    )

    unsupervised_model.save_model(save_path)
    print(f"Model saved at: {save_path}")


if __name__ == '__main__':
    for i in range(1, 6):
        for normalized in [True, False]:
            data_path = os.path.join("data", f"clean_data_{i}.csv")
            normalized_label = "normalized" if normalized else "unnormalized"
            tabnet_pretrain(data_path=data_path,
                            normalize_data=normalized,
                            save_path=os.path.join("models", f"pretrained_{normalized_label}_{i}"),
                            )
