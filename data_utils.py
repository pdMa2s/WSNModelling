import tensorflow as tf
from keras.backend import mean, sum
from numpy import array, concatenate
from pandas import read_csv
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow_core import Tensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split_dataset(data, n_test_entries):
    # split into standard weeks
    train, test = data[:-n_test_entries], data[-n_test_entries:]

    return train, test


def scale_data(dataset: array, scaler: TransformerMixin) -> array:
    assert scaler is not None and dataset is not None

    return scaler.fit_transform(dataset)


def standardize_data(dataset: array):
    cs = StandardScaler()
    return scale_data(dataset, cs), cs


def unscale_data(x: array, y: array, scaler, axis: int = 1) -> array:
    assert x is not None and y is not None and scaler is not None
    assert axis is not None and isinstance(axis, int)

    conc_data = concatenate((x, y), axis=axis)
    return scaler.inverse_transform(conc_data)


def train_test_val_split(x: array, y: array, test_size: float = .2, val_size: float = .1):
    assert test_size > 0 and val_size > 0

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size, shuffle=False)
    return x_train, y_train, x_test, y_test, x_val, y_val


def nash_sutcliffe(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Computes the Nash-Sutcliffe Coefficient between two vectors.

    If all observed values are equal to the observed mean, with estimation equal to observation, returns 1.
    If all observed values are equal to the observed mean, with estimation different than observation, returns 0.

    :param y_true: Estimated data
    :param y_pred: Observed data
    :return: Root Mean Squared Error
    :raises:
    AssertionError: if vector_x and vector_y are not of type list
    AssertionError: if the elements of vector_x and vector_y are not of the type float
    AssertionError: if vector_x and vector_y are not of the same length
    """
    if not isinstance(y_true, Tensor):
        y_true = tf.convert_to_tensor(y_true, np.float32)
    if not isinstance(y_pred, Tensor):
        y_pred = tf.convert_to_tensor(y_pred, np.float32)

    true_mean = mean(y_true)

    numerator = sum((y_pred - y_true)**2)
    denominator = sum((y_true - true_mean)**2)

    return - 1 * (1 - tf.math.divide_no_nan(numerator, denominator))


def load_fontinha():
    dataset = read_csv('data/simDataFontinha.csv', header=0)
    dataset['hFfin'] = dataset['hFfin'] - dataset['hFini']
    return dataset.drop(['E', 'hFini', 'agrg'], axis=1), 1, (1, 3), 24


def load_richmond():
    dataset = read_csv('data/simDataRichmond.csv', header=0)
    final_level_columns = [_ for _ in dataset.columns if _.startswith("hFf")]
    for index, column_name in enumerate(final_level_columns):
        dataset[f"hFi{index}"] = dataset[column_name] - dataset[f"hFi{index}"]
    columns_to_remove = ['E', 'agrg']
    columns_to_remove.extend(final_level_columns)
    return dataset.drop(columns_to_remove, axis=1), 6, (6, 47), 24


def load_adcl():
    dataset = read_csv('data/adcl_data_valve_15min.csv', header=0, index_col='Time', parse_dates=True)
    abs_levels = pd.DataFrame()

    for col in [_ for _ in dataset.columns if _.startswith("Res_")]:
        abs_levels[col] = dataset[col]
        dataset[col] = dataset[col] - dataset[col].shift()
    dataset.dropna(inplace=True)
    abs_levels.dropna(inplace=True)
    return dataset, 3, (3, 8), 192, abs_levels.iloc[1:]


def load_adcl_raw():
    adcl_raw = pd.read_csv("data/adcl_data.csv")
    adcl_raw = adcl_raw.dropna()
    adcl_raw["Time"] = pd.to_datetime(adcl_raw['Time'], utc=True)
    return adcl_raw.set_index('Time')


def plot_results_bars(_real: np.ndarray, _pred: np.ndarray):
    assert len(_real) == len(_pred)
    bar_width = 0.35

    for i, _data in enumerate(zip(_real.T, _pred.T)):
        r, p = _data
        bar_index = np.arange(len(r))
        plt.title(f"Tank{i}")
        plt.grid()
        plt.bar(bar_index, r, bar_width, label='real')
        plt.bar(bar_index+bar_width, p, bar_width, label='predicted')
        plt.ylabel('Tank level deviation')
        plt.legend()
        plt.show()


def plot_results_lines(_real: np.ndarray, *_preds: np.ndarray, titles: list = None):

    for i, real_data in enumerate(_real.T):
        plt.title(f"Tank {titles[i]}") if titles is not None else plt.title(f"Tank {i}")
        plt.grid()
        plt.plot(real_data, label='real')
        for p_i, pr in enumerate(_preds):
            assert len(real_data) == len(pr.T[i])
            plt.plot(pr.T[i], label=f'predicted{p_i}')

        plt.ylabel('Tank level deviation')
        plt.legend()
        plt.show()


def calculate_abs_levels(intial_levels: np.ndarray, y_pred: np.ndarray):
    intial_levels = intial_levels.copy()
    if intial_levels.ndim == 1:
        intial_levels = intial_levels.reshape((1, -1))
    intial_and_variations = np.append(intial_levels, y_pred, axis=0)

    return intial_and_variations.cumsum(axis=0)[1:]