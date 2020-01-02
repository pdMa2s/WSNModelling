from time import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import plot_partial_dependence
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator
from data_processor import process_data_pipeline, \
    process_differential_column


def show_partial_dependence_plot(estimator: BaseEstimator, x_train, _features, _feature_names=None):
    print('Computing partial dependence plots...')
    _tic = time()
    # We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
    # with the brute method.
    plot_partial_dependence(estimator, x_train, _features, target=0, feature_names=_feature_names,
                            n_jobs=-1, grid_resolution=100, n_cols=6)
    print("done in {:.3f}s".format(time() - _tic))
    fig = plt.gcf()
    fig.suptitle('Partial dependence of house value on non-location features\n'
                 'for the California housing dataset, with MLPRegressor')
    fig.subplots_adjust(hspace=0.3)
    fig.show()


if __name__ == '__main__':
    fontinha_data = pd.read_csv("fontinha_data.csv")
    fontinha_differential_data = process_differential_column(fontinha_data.values, [0], [1])

    richmond_data = pd.read_csv("richmond_data.csv")
    richmond_differential_data = process_differential_column(richmond_data.values, [_ for _ in range(6)],
                                                             [_ for _ in range(6, 12)])

    for config in [
        # {'data': fontinha_data.values, 'target_indexes': [0],
        #  'feature_names': ['hIni', 'dmd0', 'dmd1', 'agrg', 'pumps']},
        # {'data': fontinha_data.drop(['agrg'], axis=1).values, 'target_indexes': [0],
        #  'feature_names': ['hIni', 'dmd0', 'dmd1', 'pumps']},
        # {'data': fontinha_differential_data, 'target_indexes': [0],
        #  'feature_names': ['dmd0', 'dmd1', 'agrg', 'pumps']},
        #{'data': richmond_data.values, 'target_indexes': [_ for _ in range(6)],
         #'feature_names': [f"hIn{_}" for _ in range(6)] + [f"dmd{_}" for _ in range(41)] + ['agrg'] + [f"pumps{_}" for _ in range(7)]},
        {'data': richmond_differential_data, 'target_indexes': [_ for _ in range(6)],
         'feature_names': [f"dmd{_}" for _ in range(41)] + ['agrg'] + [f"pumps{_}" for _ in range(7)]}
                   ]:

        X_train, X_test, y_train, y_test = process_data_pipeline(config['data'], config['target_indexes'], 96)
        print("Training MLPRegressor...")
        tic = time()
        est = make_pipeline(QuantileTransformer(),
                            MLPRegressor(hidden_layer_sizes=(50, 50),
                                         learning_rate_init=0.001, learning_rate='adaptive',
                                         early_stopping=True))
        est.fit(X_train, y_train)
        print("done in {:.3f}s".format(time() - tic))
        print("Test R2 score: {:.2f}".format(est.score(X_test, y_test)))

        show_partial_dependence_plot(est, X_train, config['feature_names'], _feature_names=config['feature_names'])