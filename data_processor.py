import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence


def split_data(y, x, test_size) -> (np.array, np.array, np.array, np.array):
    split_index = len(x) - test_size-1
    return x[:split_index, :], x[split_index:, :], y[:split_index, :], y[split_index:, :]


def split_features_and_target(_data: np.ndarray, target_indexes: [int]) -> (np.array, np.array):
    return np.array([_data[:, idx] for idx in target_indexes]).transpose(), np.delete(_data, target_indexes, axis=1)


def calculate_column_variance(_data: np.array, /, index_col1, index_col2):
    return _data[:, index_col1] - _data[:, index_col2]


def show_partial_dependence_plot(estimator, x_train, _features, _feature_names=None):
    print('Computing partial dependence plots...')
    _tic = time()
    # We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
    # with the brute method.
    plot_partial_dependence(estimator, x_train, _features, feature_names=_feature_names,
                            n_jobs=8, grid_resolution=40)
    print("done in {:.3f}s".format(time() - _tic))
    fig = plt.gcf()
    fig.suptitle('Partial dependence of house value on non-location features\n'
                 'for the California housing dataset, with MLPRegressor')
    fig.subplots_adjust(hspace=0.3)
    fig.show()

