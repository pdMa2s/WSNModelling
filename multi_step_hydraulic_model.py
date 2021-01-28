import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Dense, Dropout

from data_utils import load_adcl, plot_results_lines, calculate_abs_levels, split_dataset
from data_utils import nash_sutcliffe, standardize_data
from simulation import adcl_simulation


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower left')
    plt.show()


def build_nn(n_outputs: int, n_features: int):
    _in = Input(shape=(n_features,))
    d = Dense(100, activation='tanh')(_in)
    d = Dropout(.1)(d)
    d = Dense(50, activation='tanh')(d)
    d = Dropout(.1)(d)
    d = Dense(20, activation='tanh')(d)
    d = Dense(10, activation='tanh')(d)
    out = Dense(n_outputs, activation='tanh')(d)

    _model = Model(inputs=_in, outputs=out)

    _model.compile(loss=nash_sutcliffe, optimizer="adam", metrics=['mae'])

    return _model, 1


def build_multi_nn(n_outputs: int, n_features: int):
    outs = []
    ins = []
    for output_n in range(n_outputs):
        column_input = Input(shape=(n_features,), name=f"in_c{output_n}")
        ins.append(column_input)
        d = Dense(50, activation='tanh')(column_input)
        d = Dropout(.1)(d)
        d = Dense(20, activation='tanh')(d)
        d = Dropout(.1)(d)
        d = Dense(10, activation='tanh')(d)
        d = Dense(1, activation='tanh', name=f"out_c{output_n}")(d)
        outs.append(d)

    conc_layer = Concatenate(axis=1)(outs)
    _model = Model(inputs=ins, outputs=conc_layer)

    _model.compile(loss=nash_sutcliffe, optimizer="adam", metrics=['mae'])

    return _model, len(ins)


if __name__ == '__main__':

    dataset, split_target_feature_index, demand_indexes, test_size, abs_levels = load_adcl()

    abs_levels_test_set = abs_levels[abs_levels.index >= abs_levels.index[-test_size-1]]

    n_steps = 1
    dataset_values = dataset.values
    dataset_values[:, demand_indexes[0]:demand_indexes[1]], scaler = \
        standardize_data(dataset_values[:, demand_indexes[0]:demand_indexes[1]])

    train, test = split_dataset(dataset_values, test_size)

    # model, n_inputs = build_multi_nn(split_train_test_index, dataset_values.shape[1]-split_train_test_index)
    model, n_inputs = build_multi_nn(split_target_feature_index, dataset_values.shape[1] - split_target_feature_index)

    history = model.fit([train[:, split_target_feature_index:] for _ in range(n_inputs)],
                        train[:, :split_target_feature_index], epochs=50,
                        use_multiprocessing=True)

    plot_loss(history)

    y_true = test[:, :split_target_feature_index]
    x_test = [test[:, split_target_feature_index:] for _ in range(n_inputs)]
    score = model.evaluate(x_test, y_true)

    print(score)
    y_pred = model.predict(x_test)
    # plot_results_bars(y_true, y_pred)
    pred_abs_levels = calculate_abs_levels(abs_levels_test_set.iloc[0].values, y_pred)
    plot_results_lines(abs_levels_test_set.iloc[1:].values, pred_abs_levels, adcl_simulation(),
                       titles=list(dataset.columns[:3]))
