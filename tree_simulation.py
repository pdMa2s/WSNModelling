from sklearn.tree import DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestRegressor
from data_utils import load_adcl, standardize_data, split_dataset, nash_sutcliffe
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


def build_decision_tree():
    _model = DecisionTreeRegressor()
    return _model


def build_random_forest():
    _model = RandomForestRegressor(n_estimators=500, n_jobs=5)
    return _model


def build_xgboost():
    _model = MultiOutputRegressor(xgb.XGBRegressor(objective="reg:squaredlogerror", max_depth=10, verbosity=2))
    return _model


if __name__ == '__main__':
    dataset, split_target_label_index, demand_indexes, test_size, abs_levels = load_adcl()

    n_steps = 1
    dataset_values = dataset.values
    dataset_values[:, demand_indexes[0]:demand_indexes[1]], scaler = \
        standardize_data(dataset_values[:, demand_indexes[0]:demand_indexes[1]])

    train, test = split_dataset(dataset_values, test_size)
    model = build_random_forest() # build_xgboost()
    model.fit(train[:, split_target_label_index:], train[:, :split_target_label_index])

    y_true = test[0, :split_target_label_index].reshape((1, -1))
    y_pred = model.predict(test[0, split_target_label_index:].reshape((1, -1)))

    nash = nash_sutcliffe(y_true, y_pred)
    print(nash.numpy())

    # plt.figure(figsize=(50, 50))
    # plot_tree(model, filled=True)
    # plt.savefig("tree.pdf")
    # plt.show()
    # export_graphviz(model, out_file='dt.dot')