import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Normalizer
from tabulate import tabulate
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from time import time
from data_processor import process_data_pipeline, process_differential_column
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


def previous_step_repeater(y_pred: np.ndarray, features: np.ndarray, i):
    if y_pred.ndim == 1:
        features[i+1, 0] = y_pred[0]
    else:
        features[i+1, 0: len(y_pred[0])] = y_pred[0]
    print()


def multi_step_simulation(estimator, _features: np.array, input_modifier=None):
    y_pred = None
    for i, x in enumerate(_features):
        y_pred = estimator.predict(x.reshape(1, -1)) if y_pred is None else \
            np.append(y_pred, estimator.predict(x.reshape(1, -1)), axis=0)
        if input_modifier is not None and i != len(_features)-1:
            input_modifier(y_pred, _features, i)
    return y_pred


def print_table(rows: [[]], header: [] = None):
    print(tabulate(rows, headers=header) if header else tabulate(rows, headers=header))


def calculate_metrics(y_true, y_pred, metrics: []):
    metric_scores = []
    for m in metrics:
        metric_scores.append(m(y_true, y_pred))
    return metric_scores


def plot_results(_real: np.ndarray, _pred: np.ndarray):
    assert len(_real) == len(_pred)
    bar_width = 0.35

    for i, _data in enumerate(zip(_real.T, _pred.T)):
        r, p = _data
        bar_index = np.arange(len(r))
        plt.title(f"Tank{i}")
        plt.bar(bar_index, r, bar_width, label='real')
        plt.bar(bar_index+bar_width, p, bar_width, label='predicted')
        plt.ylabel('Tank level deviation')
        plt.legend()
        plt.show()


def evaluate_multi_step(estimator, _features: np.array, _targets: np.array, n_steps: int, input_modifier=None,
                        plot_func=None):
    assert len(_features) % n_steps == 0 and len(_targets) % n_steps == 0
    assert len(_features) == len(_targets)

    y_pred = None
    for t in range(0, len(_features), n_steps):
        y_pred = np.append(y_pred, multi_step_simulation(estimator, _features[t:t+n_steps, :], input_modifier), axis=0)\
            if y_pred is not None else multi_step_simulation(estimator, _features[t:t+n_steps, :], input_modifier)

    assert len(y_pred) == len(_targets)
    metric_scores = calculate_metrics(_targets, y_pred, [r2_score, mean_absolute_error])

    print(f"Multi-step simulation scores:\nr2: {metric_scores[0]}\nmae: {metric_scores[1]}")
    if plot_func is not None:
        plot_func(_targets[-24:, :], y_pred[-24:, :] if y_pred.ndim > 1 else y_pred.reshape(-1, 1)[-24:, :])


def calc_valves_from_tanks(dataframe):
    valves = dataframe.copy()
    valves['dH'] = valves['Res_Espinheira'] - valves['Res_Espinheira'].shift().fillna(0)
    valves['Val_Espinheira'] = valves['dH'].where(valves['dH'] > 0, 0)
    valves['Val_Espinheira'] = valves['Val_Espinheira'].where(valves['Val_Espinheira'] <= 0, 1)
    valves.drop(['Time', 'Res_Espinheira', 'Res_Aveleira', 'Res_Albarqueira', 'dH'], inplace=True, axis=1)
    return valves


def calc_level_variation(dataframe):
    variations = dataframe.copy()
    variations['Time'] = pd.to_datetime(variations['Time'])
    variations['deltaT'] = (variations['Time'] - variations['Time'].shift())
    for col in dataframe.columns:
        if col not in ['Time', 'deltaT']:
            variations[col] = (dataframe[col] - dataframe[col].shift()) / (variations['deltaT'] / pd.Timedelta('1H'))
    variations.drop(['Time', 'deltaT'], inplace=True, axis=1)
    variations.dropna(inplace=True)
    return variations


def filter_tanks_data(dataframe):
    for col in dataframe.columns:
        if col != 'Time':
            dataframe[col].interpolate('time')
    return dataframe


if __name__ == '__main__':
    # testing_entry = {'data': None, 'input_modifier': None}
    data_dir = "dataGeneration/"
    demands_data = pd.read_csv(f'{data_dir}demands_hyd_ann.csv', sep=';')
    tanks_data = pd.read_csv(f'{data_dir}tanks_hyd_ann.csv', sep=';')
    pumps_data = pd.read_csv(f'{data_dir}pumps_hyd_ann.csv', sep=';')
    valves_data = calc_valves_from_tanks(tanks_data)


    input_data = pd.concat((demands_data, pumps_data), axis=1)
    input_data.drop('Time', inplace=True, axis=1)
    target_data = calc_level_variation(tanks_data)

    for col in input_data.columns:
        input_data[col].interpolate(inplace=True)
    for col in target_data.columns:
        target_data[col].interpolate(inplace=True)
    target_data = target_data.diff().fillna(0)

    adcl_data = np.append(target_data.values, input_data.values, axis=1)
    print()

    fontinha_data = pd.read_csv("dataGeneration/fontinha_data.csv")
    fontinha_differential_data = process_differential_column(fontinha_data.values, [0], [1])

    richmond_data = pd.read_csv("dataGeneration/richmond_data_1h.csv")
    richmond_differential_data = process_differential_column(richmond_data.values, [_ for _ in range(6)],
                                                             [_ for _ in range(6, 12)])

    test_set_size = 48
    for config in [
       # {'data': fontinha_data.values, 'target_indexes': [0], 'input_modifier': previous_step_repeater},
       # {'data': data.drop(['agrg'], axis=1).values, 'target_indexes': [0], 'input_modifier': None},
       # {'data': fontinha_differential_data, 'target_indexes': [0], 'input_modifier': None},
       # {'data': richmond_data.values, 'target_indexes': [_ for _ in range(6)], 'input_modifier': previous_step_repeater},
        {'data': richmond_differential_data, 'target_indexes': [_ for _ in range(6)], 'input_modifier': None},
       #{'data': adcl_data, 'target_indexes': [0, 1, 2], 'input_modifier': None}
    ]:

        X_train, X_test, y_train, y_test = process_data_pipeline(config['data'], config['target_indexes'],
                                                                 test_set_size)
        print("Training MLPRegressor...")
        tic = time()
        est = make_pipeline(Normalizer(),
                            MLPRegressor(hidden_layer_sizes=(100, ), activation='logistic',
                                         learning_rate_init=0.001, learning_rate='adaptive', n_iter_no_change=20,
                                         verbose=True))

        est.fit(X_train, y_train)
        evaluate_multi_step(est, X_test, y_test, 24, plot_func=plot_results, input_modifier=config['input_modifier'])