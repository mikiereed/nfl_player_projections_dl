import numpy as np
from typing import Optional


def load_dataset(csv_path: str, feature_columns: Optional[list] = None, target_columns: Optional[list] = None, add_intercept=False):
    """
    Loads a csv dataset into feature and target numpy arrays, and feature and target column name lists
    If feature_columns = None, feature columns will be set with any columns that start with "Feature"
    If target_columns = None, target columns will be set with any columns that start with "Target"
    :param csv_path: absolute path to csv dataset
    :param feature_columns: list of feature columns to use specifically
    :param target_columns: list of target columns to use specifically
    :param add_intercept: add an intercept column to feature array
    :return: feature and target numpy arrays, and feature and target column name lists
    """
    # if feature_columns is None or target_columns is None:
    with open(csv_path, 'r', newline='') as f:
        columns = f.readline().strip().split(',')

    if feature_columns is None:
        feature_columns = [column for column in columns if column.startswith("Feature")]

    feature_columns_indices = [columns.index(column) for column in feature_columns]

    if target_columns is None:
        target_columns = [column for column in columns if column.startswith("Target")]

    target_columns_indices = [columns.index(column) for column in target_columns]

    features = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=feature_columns_indices)
    targets = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=target_columns_indices)

    if features.ndim == 1:
        features = np.expand_dims(features, -1)

    if add_intercept:
        features = add_intercept_fn(features)

    return features, targets, feature_columns, target_columns


def add_intercept_fn(x):
    x_with_intercept = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    x_with_intercept[:, 0] = 1
    x_with_intercept[:, 1:] = x

    return x_with_intercept
