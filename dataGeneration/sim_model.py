import math
import random

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
from testbed_utils import scale_data, print_metrics, split_data, print_cross_val_metrics, plot_validation, inverse_data, \
    calculate_metrics, to_supervised_index, calculate_predicted_metrics, plot_lines, plot_bars


# splits the data using an method provided by scikit learn library, the sets are random
def split_data_sklearn(x, y, test_size=.20, n_observations=24):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=n_observations/len(x_test))
    return x_train, x_test, x_val, y_train, y_test, y_val


# split the data by simply dividing the sets
def split_data_utils(x, y, n_observations=24):
    test_split = int(.80 * (len(x) / n_observations)) * n_observations
    x_train, x_test, x_val = split_data(x, [test_split - n_observations, len(x) - n_observations])
    y_train, y_test, y_val = split_data(y, [test_split - n_observations, len(y) - n_observations])
    return x_train, x_test, x_val, y_train, y_test, y_val


def search_params(x, y):
    params = {'hidden_layer_sizes': [(35,), (20,), (100,), (20, 10)],
              'activation': ['relu'],
              'solver': ['adam'], 'batch_size': [75, 100, 150],
              'learning_rate': ['constant', 'invscaling'],
              'learning_rate_init': [.0025]}
    clf = GridSearchCV(MLPRegressor(), params, cv=3, n_jobs=6, scoring='neg_mean_absolute_error')
    clf.fit(x, y)
    return clf.best_params_, clf.best_score_


def param_validation(model, x, y):
    #plot_validation(model, x, y, 'batch_size', [75, 100, 200, 300])
    #plot_validation(model, x, y, 'learning_rate_init', [.001, .002, .0025, .003])
    #plot_validation(model, x, y, 'activation', ['identity', 'logistic', 'tanh', 'relu'])
    #plot_validation(model, x, y, 'hidden_layer_sizes', [(70,), (40,), (22,), (20, 10)])
    #plot_validation(model, x, y, 'solver', ['lbfgs', 'adam', 'sgd'])
    plot_validation(model, x, y, 'learning_rate', ['adaptive', 'constant', 'invscaling'])
    #plot_validation(model, x, y, 'max_iter', [50, 100, 150, 200])


def process_data_sets(data, label_sep_idex, test_size=.20, n_observations=24):

    scl_data, scaler = scale_data([data])
    scl_data = scl_data[0]
    x, y = to_supervised_index(scl_data, label_sep_indx=label_sep_idex)

    #x_train, x_test, x_val, y_train, y_test, y_val = \
     #   split_data_sklearn(x, y, test_size=test_size, n_observations=n_observations)
    x_train, x_test, x_val, y_train, y_test, y_val = split_data_utils(x, y, n_observations=n_observations)

    return x_train, x_test, x_val, y_train, y_test, y_val, scaler


def add_noise(dataset, percentage=.1):
    for r in range(len(dataset)):
        for c in range(len(dataset[r])):
            value = dataset[r][c]
            error = value * percentage
            noise = random.uniform(error * (-1), error)
            dataset[r][c] = value + noise
    return dataset


def plot_results(real, pred, sep_index):
    plt.xticks(np.append(np.arange(0, len(real), step=6), len(pred)-1))
    for i in range(sep_index - 1):
        plot_lines(real[:, i], pred[:, i], 'Tank ' + str(i), x_label='Time (h)', y_label='tank level (m)')

    if len(real.shape) == 2:
        plt.xticks(np.append(np.arange(0, len(real), step=6), len(pred)-1))
        plot_bars(real[:, sep_index - 1], pred[:, sep_index - 1], 'Used energy', x_label='Time (h)', y_label='kilowatts')
    else:
        plt.xticks(np.append(np.arange(0, len(real), step=6), len(pred)-1))
        plot_bars(real, pred, 'Used energy', x_label='Time (h)', y_label='kilowatts')


def calculate_absolute_metrics(rgr, x, y, scaler, output_sep=1):
    pred = rgr.predict(x)
    iv_real = inverse_data(scaler, y, x)
    iv_pred = inverse_data(scaler, pred, x)
    iv_real = to_supervised_index(iv_real, label_sep_indx=sep)[1]
    iv_pred = to_supervised_index(iv_pred, label_sep_indx=sep)[1]
    iv_real_tank = iv_real[:, :output_sep]
    iv_real_energy = iv_real[:, output_sep:]
    iv_pred_tank = iv_pred[:, :output_sep]
    iv_pred_energy = iv_pred[:, output_sep:]
    tank_metrics = calculate_predicted_metrics(iv_real_tank, iv_pred_tank)
    energy_metrics = calculate_predicted_metrics(iv_real_energy, iv_pred_energy)
    return tank_metrics, energy_metrics


def add_noise_training_set(set, noise_percentage=0.0, set_percentage=.3):
    split_index = int(len(set) * set_percentage)
    noisy_set = set[split_index:]
    noisy_set = add_noise(noisy_set, noise_percentage)
    set[split_index:] = noisy_set
    return set


def plot_features_richmond(features):

    tanks = features[:, :3]
    demands = features[:, -10:-7]
    pumps = features[:, -2:]
    transposed = np.concatenate((tanks, demands, pumps), axis=1)
    transposed = transposed.transpose()
    titles_and_labels = [('Initial height 0', 'meters'), ('Initial height 1', 'meters'), ('Initial height 2', 'meters'),
                         ('Flow demand 0', 'm\u00b3\\h'), ('Flow demand 1', 'm\u00b3\\h'),
                         ('Aggregated demand', 'm\u00b3\\h'), ('Pump status 6', 'Pump status'), ('Pump status 7', 'Pump status')]
    for i in range(len(transposed)):
        plt.title(titles_and_labels[i][0])
        plt.xlabel('Time (hrs)')
        plt.ylabel(titles_and_labels[i][1])
        plt.xticks(np.append(np.arange(0, len(transposed[i]), step=3), len(transposed[i])-1))

        if i >= len(transposed) - 2:
            y_ticks = np.append(np.arange(0, math.ceil(max(transposed[i])), step=.2), math.ceil(max(transposed[i])))
            plt.yticks(y_ticks)
            #plt.ylim(0, math.ceil(max(transposed[i])))
            plt.step([x for x in range(len(transposed[i]))], transposed[i], color='magenta')
        else:
            plt.bar([t for t in range(len(transposed[i]))], transposed[i], color='magenta')

        plt.show()


def plot_features_fontinha(features):
    transposed = features.transpose()
    titles_and_labels = [('Initial height', 'meters'), ('Flow demand VC', 'm\u00b3\\h'), ('Flow demand R', 'm\u00b3\\h'),
                         ('Aggregated demand', 'm\u00b3\\h'), ('Pump', 'Pump status')]
    for i in range(len(transposed)):
        plt.title(titles_and_labels[i][0])
        plt.xlabel('Time (h)')
        plt.ylabel(titles_and_labels[i][1])
        plt.xticks(np.append(np.arange(0, len(transposed[i]), step=3), len(transposed[i])-1))

        if i >= len(transposed) - 1:
            y_ticks = np.append(np.arange(0, math.ceil(max(transposed[i])), step=.2), math.ceil(max(transposed[i])))
            plt.yticks(y_ticks)
            plt.ylim(0, math.ceil(max(transposed[i])))
            plt.step([x for x in range(len(transposed[i]))], transposed[i], color='magenta')
        else:
            plt.bar([t for t in range(len(transposed[i]))], transposed[i], color='magenta')

        plt.show()


if __name__ == '__main__':
    # data = pd.read_csv('../../data/simData.csv')
    richmond_data = pd.read_csv('../../data/simDataRichmond.csv')
    pump_e = richmond_data.loc[:, ['E']]
    pump_stat = richmond_data.loc[:, 'agrg': 'pum6']
    pump_data = pd.concat([pump_e, pump_stat], axis=1)
    #train_amount = [.20, .80, .90, .95, .97, .98, .99, .999]
    regressors_and_data = {

        #'Fontinha': (pd.read_csv('../../data/simDataFontinha.csv'), 2, [.20],
         #            MLPRegressor(solver='adam', activation='relu', learning_rate_init=.002, hidden_layer_sizes=(15,),
          #                        batch_size=50, learning_rate='constant'), True, [0], 1, False, 1,
           #          plot_features_fontinha, 1),
        'Richmond': (richmond_data, 7, [.20],
                     MLPRegressor(batch_size=100, activation='relu', learning_rate='constant', learning_rate_init=.0025,
                                  hidden_layer_sizes=(22,), solver='adam'),
                     True, [0], 1, False, 6, plot_features_richmond, 6),
        #'Richmond_pumps': (pump_data, 1, MLPRegressor(hidden_layer_sizes=(40,), activation='relu',
         #                                             learning_rate_init=.002, batch_size=250)), True, [0], 1, False),

    }

    for data_name, params in regressors_and_data.items():
        data, sep, test_splits, rgr, plot_individual, noise, repeat, avg_plot, output_sep, plot_func, tanks_sep = \
            params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], \
            params[9], params[10]

        for s in test_splits:

            for tn in noise:

                for n in noise:
                    tank_metric_sum = [0, 0, 0, 0]
                    energy_metric_sum = [0, 0, 0, 0]
                    for r in range(repeat):
                        x_train, x_test, x_val, y_train, y_test, y_val, scaler = \
                            process_data_sets(data, sep, test_size=s, n_observations=24)
                        print("training set size: ", len(x_train))
                        # search for the hyperparameters that obtain the best results
                        # print(search_params(x_train, y_train))
                        # plot the validation scores for several parameters
                        # param_validation(rgr, x_train, y_train)

                        x_train = add_noise_training_set(x_train, noise_percentage=tn)
                        rgr.fit(x_train, y_train)

                        pred = rgr.predict(x_test)
                        x_test = add_noise(x_test, percentage=n)
                        x_val = add_noise(x_val, percentage=n)

                        # print scores for several metrics for the validation set
                        #mse, mae, s2 = calculate_metrics(rgr, x_val, y_val)
                        tank_metrics, energy_metrics = calculate_absolute_metrics(rgr, x_val, y_val, scaler, output_sep=output_sep)

                        tank_metric_sum = [sum(x) for x in zip(tank_metrics, tank_metric_sum)]
                        energy_metric_sum = [sum(x) for x in zip(energy_metrics, energy_metric_sum)]

                        if plot_individual:
                            # prints cross validation scores with 4 folds and neg mean absolute error for the test set
                            print_cross_val_metrics(rgr, x_test, y_test, set_name='test')
                            print_metrics(tank_metrics[0], tank_metrics[1], tank_metrics[2], set_name='val tank')
                            print_metrics(energy_metrics[0], energy_metrics[1], energy_metrics[2], set_name='val energy')

                            pred_val = []
                            for t in range(len(x_val)):
                                yhat = rgr.predict(x_val[t].reshape(1, -1))
                                pred_val.append(yhat[0])
                                if t < len(pred_val) - 1:
                                    x_val[t+1][:tanks_sep] = yhat[0][:tanks_sep]


                            #pred_val = rgr.predict(x_val)
                            iv_pred = to_supervised_index(inverse_data(scaler, np.array(pred_val), x_val), label_sep_indx=sep)
                            iv_pred_y = iv_pred[1]
                            iv_pred_x = iv_pred[0]
                            iv_val = to_supervised_index(inverse_data(scaler, y_val, x_val), label_sep_indx=sep)
                            iv_val_y = iv_val[1]
                            iv_val_x = iv_val[0]
                            plot_func(iv_val_x)
                            plot_results(iv_val_y, iv_pred_y, sep)
                    if avg_plot:
                        print("\ntraining noise: ", tn)
                        print("\ntesting noise: ", n)
                        tank_metrics_average = [a / repeat for a in tank_metric_sum]
                        energy_metrics_average = [a / repeat for a in energy_metric_sum]

                        print_metrics(tank_metrics_average[0], tank_metrics_average[1], tank_metrics_average[2], set_name='tank average')
                        print_metrics(energy_metrics_average[0], energy_metrics_average[1], energy_metrics_average[2], set_name='energy average')
