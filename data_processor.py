import numpy as np
import pandas as pd


def split_data(y, x, test_size) -> (np.array, np.array, np.array, np.array):
    split_index = len(x) - test_size
    return x[:split_index, :], x[split_index:, :], y[:split_index, :], y[split_index:, :]


def split_features_and_target(_data: np.ndarray, target_indexes: [int]) -> (np.array, np.array):
    return np.array([_data[:, idx] for idx in target_indexes]).transpose(), np.delete(_data, target_indexes, axis=1)


def calculate_column_variance(_data: np.array, index_col1, index_col2):
    return _data[:, index_col1] - _data[:, index_col2]


def process_data_pipeline(data, target_indexes, test_set_size):
    targets, features = split_features_and_target(data, target_indexes)
    X_train, X_test, y_train, y_test = split_data(targets, features, test_set_size)
    return X_train, X_test, y_train, y_test


def process_differential_column(_data: np.array, right_columns: list, left_columns: list):
    differential_data = _data.copy()
    for right, left in zip(right_columns, left_columns):
        variance_column = calculate_column_variance(_data, right, left)
        differential_data[:, right] = variance_column
    differential_data = np.delete(differential_data, left_columns, axis=1)
    return differential_data


def calc_valves_from_tanks(dataframe):
    valves = dataframe.copy()
    valves['dH'] = valves['Res_Espinheira'] - valves['Res_Espinheira'].shift().fillna(0)
    valves['Val_Espinheira'] = valves['dH'].where(valves['dH'] > 0, 0)
    valves['Val_Espinheira'] = valves['Val_Espinheira'].where(valves['Val_Espinheira'] <= 0, 1)
    valves.drop(['dH'], axis=1, inplace=True)
    return valves


def calc_level_variation(dataframe):
    variations = dataframe.copy()
    variations['Time'] = pd.to_datetime(variations['Time'])
    variations.set_index('Time', drop=False, inplace=True)
    variations['deltaT'] = (variations['Time'] - variations['Time'].shift())
    for col in dataframe.columns:
        if col.startswith('Res'):
            variations[col] = (variations[col] - variations[col].shift()) * (variations['deltaT'] / pd.Timedelta('1H'))
    variations.drop(['deltaT'], inplace=True, axis=1)
    variations.dropna(inplace=True)
    return variations


def filter_tanks_data(dataframe):
    dataframe['Time'] = pd.to_datetime(dataframe['Time'])
    dataframe.set_index('Time', drop=False, inplace=True)
    for col in dataframe.columns:
        if col != 'Time':
            dataframe[col].interpolate('time', inplace=True)
    return dataframe


def pumps_to_bool(dataframe):
    for col in dataframe.columns:
        if col.startswith('P_'):
            dataframe[col] = dataframe[col].mask(dataframe[col] > 0.5, 1)
            dataframe[col] = dataframe[col].mask(dataframe[col] <= 0.5, 0)
    return dataframe


def flow_to_volume(dataframe):

    for col in dataframe.columns:
        if col.startswith('PE_'):
            dataframe[col] = dataframe[col] * (dataframe['deltaT'] / pd.Timedelta('1H'))
    return dataframe


def volume_to_flow(dataframe):
    for col in dataframe.columns:
        if col.startswith('PE_'):
            dataframe[col] = dataframe[col] * (pd.Timedelta('1H') / dataframe['deltaT'])
    return dataframe


def agg_unique_controls(dataframe):
    controls = dataframe.copy()
    controls.dropna(axis=0, inplace=True)
    controls.drop(['deltaT'], axis=1, inplace=True)
    controls['code'] = ''
    aggregators = {}
    for col in dataframe.columns:
        if col.startswith('Res_') or col == 'Time':
            aggregators[col] = 'first'
        elif col.startswith('PE_'):
            aggregators[col] = 'mean'
        elif col.startswith('P_') or col.startswith('Val_'):
            aggregators[col] = 'first'
            controls[col] = controls[col].astype(int).astype(str)
            controls['code'] += controls[col]
    controls['regions'] = (controls['code'] != controls['code'].shift(1)).astype(int).cumsum()
    grouped = controls.groupby('regions')
    grouped = grouped.agg(aggregators)
    controls.drop(['code'], axis=1, inplace=True)

    return grouped


if __name__ == '__main__':
    data_dir = "dataGeneration/"
    raw_data = pd.read_csv(f'{data_dir}adcl_data.csv', sep=';')

    in_progress_data = filter_tanks_data(raw_data)
    in_progress_data = calc_valves_from_tanks(in_progress_data)
    in_progress_data = pumps_to_bool(in_progress_data)
    in_progress_data['deltaT'] = in_progress_data['Time'].diff()
    in_progress_data = flow_to_volume(in_progress_data)
    in_progress_data = agg_unique_controls(in_progress_data)
    in_progress_data['deltaT'] = in_progress_data['Time'].diff()
    in_progress_data = volume_to_flow(in_progress_data)
    in_progress_data = calc_level_variation(in_progress_data)

    in_progress_data.drop('Time', axis=1).to_csv(f'{data_dir}adcl_grouped_data.csv', index=False)
    print('Finishing...')