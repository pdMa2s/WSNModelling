import numpy as np


def split_data(y, x, test_size) -> (np.array, np.array, np.array, np.array):
    split_index = len(x) - test_size
    return x[:split_index, :], x[split_index:, :], y[:split_index, :], y[split_index:, :]


def split_features_and_target(_data: np.ndarray, target_indexes: [int]) -> (np.array, np.array):
    return np.array([_data[:, idx] for idx in target_indexes]).transpose(), np.delete(_data, target_indexes, axis=1)


def calculate_column_variance(_data: np.array, /, index_col1, index_col2):
    return _data[:, index_col1] - _data[:, index_col2]


def process_data_pipeline(data, target_indexes, test_set_size):
    targets, features = split_features_and_target(data, target_indexes)
    X_train, X_test, y_train, y_test = split_data(targets, features, test_set_size)
    return X_train, X_test, y_train, y_test


def process_differential_column(_data: np.array, right_columns: list, left_columns: list):
    differential_data = _data.copy()
    for right, left in zip(right_columns, left_columns):
        variance_column = calculate_column_variance(_data, right, left)
        differential_data[:, right] = variance_column
        differential_data = np.delete(differential_data, left, axis=1)
    return differential_data



