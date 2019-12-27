import pandas as pd
from data_processor import split_features_and_target, split_data, calculate_column_variance
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence


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


if __name__ == '__main__':
    data = pd.read_csv("fontinha_data.csv")
    variance_column = calculate_column_variance(data.values, 0, 1)
    variance_data = data.values.copy()
    variance_data[:, 0] = variance_column
    variance_data = np.delete(variance_data, 1, axis=1)

    for data_values, features_names in [(data.values, ['hIni', 'dmd0', 'dmd1', 'agrg', 'pumps']),
                                        (data.drop(['agrg'], axis=1).values, ['hIni', 'dmd0', 'dmd1', 'pumps']),
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
        print("done in {:.3f}s".format(time() - tic))
        print("Test R2 score: {:.2f}".format(est.score(X_test, y_test)))

        show_partial_dependence_plot(est, X_train, features_names, _feature_names=features_names)