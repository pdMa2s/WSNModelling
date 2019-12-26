import pandas as pd
from time import time
from data_processor import split_features_and_target, split_data, show_partial_dependence_plot, \
    calculate_column_variance
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
import numpy as np

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