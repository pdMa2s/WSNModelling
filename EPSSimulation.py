import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer
from tabulate import tabulate
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from time import time
from data_processor import calculate_column_variance, split_features_and_target, split_data
from sklearn.pipeline import make_pipeline


def multi_step_simulation(estimator, _features: np.array, input_modifier=None):
    y_pred = np.array([])
    for i, x in enumerate(_features):
        if input_modifier is not None:
            input_modifier(x, i)
        y_pred = np.append(y_pred, [estimator.predict(x.reshape(1, len(x)))])
    return y_pred


def print_table(rows: [[]], header: [] = None):
    print(tabulate(rows, headers=header) if header else tabulate(rows, headers=header))


def calculate_metrics(y_true, y_pred, metrics: []):
    metric_scores = []
    for m in metrics:
        metric_scores.append(m(y_true, y_pred))
    return metric_scores


def evaluate_multi_step(estimator, _features: np.array, _targets: np.array, n_steps: int, input_modifier=None):
    assert len(_features) % n_steps == 0 and len(_targets) % n_steps == 0
    assert len(_features) == len(_targets)

    y_pred = np.array([])
    for t in range(0, len(_features), n_steps):
        y_pred = np.append(y_pred, multi_step_simulation(estimator, _features[t:t+n_steps, :], input_modifier))

    assert len(y_pred) == len(_targets)
    metric_scores = calculate_metrics(_targets.reshape((len(_targets))), y_pred, [r2_score, mean_absolute_error])

    print(f"Multi-step simulation scores:\nr2: {metric_scores[0]}\nmae: {metric_scores[1]}")


if __name__ == '__main__':
    data = pd.read_csv("fontinha_data.csv")
    variance_column = calculate_column_variance(data.values, 0, 1)
    variance_data = data.values.copy()
    variance_data[:, 0] = variance_column
    variance_data = np.delete(variance_data, 1, axis=1)

    for data_values, features_names in [#(data.values, ['hIni', 'dmd0', 'dmd1', 'agrg', 'pumps']),
                                        #(data.drop(['agrg'], axis=1).values, ['hIni', 'dmd0', 'dmd1', 'pumps']),
                                        (variance_data, ['dmd0', 'dmd1', 'agrg', 'pumps'])]:
        targets, features = split_features_and_target(data_values, [0])
        X_train, X_test, y_train, y_test = split_data(targets, features, 96)
        print("Training MLPRegressor...")
        tic = time()
        est = make_pipeline(QuantileTransformer(),
                            MLPRegressor(hidden_layer_sizes=(50, 50),
                                         learning_rate_init=0.01,
                                         early_stopping=True))
        est.fit(X_train, y_train.reshape((len(y_train))))
        evaluate_multi_step(est, X_test, y_test, 24)