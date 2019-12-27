import numpy as np


def split_data(y, x, test_size) -> (np.array, np.array, np.array, np.array):
    split_index = len(x) - test_size
    return x[:split_index, :], x[split_index:, :], y[:split_index, :], y[split_index:, :]


def split_features_and_target(_data: np.ndarray, target_indexes: [int]) -> (np.array, np.array):
    return np.array([_data[:, idx] for idx in target_indexes]).transpose(), np.delete(_data, target_indexes, axis=1)


def calculate_column_variance(_data: np.array, /, index_col1, index_col2):
    return _data[:, index_col1] - _data[:, index_col2]




