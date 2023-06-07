import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 93


def get_columns(data_path: str) -> tuple:
    with open(data_path, 'r', newline='') as f:
        columns = f.readline().strip().split(',')

    feature_columns = [column for column in columns if column.startswith("Feature")]
    target_columns = [column for column in columns if column.startswith("Target")]

    feature_columns_indices = [columns.index(column) for column in feature_columns]
    target_columns_indices = [columns.index(column) for column in target_columns]

    return feature_columns, target_columns, feature_columns_indices, target_columns_indices

def load_dataset(data_path: str, add_intercept: bool = False):
    """
    Loads a csv dataset into feature and target numpy arrays, and feature and target column name lists
    If feature_columns = None, feature columns will be set with any columns that start with "Feature"
    If target_columns = None, target columns will be set with any columns that start with "Target"
    :param data_path: absolute path to csv dataset
    :param feature_columns: list of feature columns to use specifically
    :param target_columns: list of target columns to use specifically
    :param add_intercept: add an intercept column to feature array
    :return: feature and target numpy arrays, and feature and target column name lists
    """
    feature_columns, target_columns, feature_columns_indices, target_columns_indices = get_columns(data_path=data_path)
    features = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=feature_columns_indices)
    targets = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=target_columns_indices)

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


def split_dataset(data_path: str, normalize_data=True) -> tuple:
    X, Y, feature_cols, target_cols = load_dataset(data_path)

    if normalize_data:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X = x_scaler.fit_transform(X)
        Y = y_scaler.fit_transform(Y)
    else:
        y_scaler = None

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=RANDOM_STATE)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=RANDOM_STATE)

    return x_train, x_val, x_test, y_train, y_val, y_test, y_scaler
