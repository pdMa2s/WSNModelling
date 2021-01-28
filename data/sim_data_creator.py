import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def calc_valves_from_tanks_adcl(dataframe):
    valves = dataframe.copy()
    threshold = 0
    valves['dH'] = valves['Res_Espinheira'] - valves['Res_Espinheira'].shift().fillna(0)
    valves['Pipe1'] = valves['dH'].where(valves['dH'] <= threshold, 1)
    valves['Pipe1'] = valves['Pipe1'].where(valves['Pipe1'] > threshold, 0)
    valves.drop(['dH'], axis=1, inplace=True)
    return valves


def pumps_to_bool(dataframe):
    dataframe = dataframe.copy()
    for col in dataframe.columns:
        if col.startswith('P_'):
            dataframe[col] = dataframe[col].mask(dataframe[col] > 0.5, 1)
            dataframe[col] = dataframe[col].mask(dataframe[col] <= 0.5, 0)
    return dataframe


def interpolate(data):
    data = data.copy()
    data.interpolate('time', inplace=True)
    return data


def resample(data: pd.DataFrame, periodicity: str = '1h') -> pd.DataFrame:
    """
    Resamples data to a fixed PERIODICITY, filling nan with time-based interpolation
    :param data: The data t be resampled. It is expected a Dataframe that as respective of each data entry as an index
    :type data: pd.Dataframe
    :param periodicity: The periodicity at which the data will be resampled.
    :type periodicity: str
    :return:
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(periodicity, str)

    filtered_dataframe = data.copy()
    return filtered_dataframe.resample(periodicity).mean().interpolate('time').bfill().ffill()


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def centered(iy, window):
    side_window = int((window - 1) / 2)
    side_window_left = side_window + ((window - 1) % 2)
    side_window_right = side_window  # To make up for the right border
    changed_index = [side_window_left + 1, len(iy) - side_window_right]

    for i in range(changed_index[0], changed_index[1]):
        window_arr = [iy[j] for j in range(i - side_window_left, i + side_window_right + 1)]
        iy[i] = np.mean(window_arr)

    return iy


def trailing(y, window=3):
    for i in range(window - 1, len(y)):
        y[i] = np.mean([y[j] for j in range(i + 1 - window, i + 1)])
    return y


def prediction(adcl, in_of_samples, iwindow, icolor, is_resampled=False):
    my_label = 'Prediction' if is_resampled else 'Prediction_raw'
    ix = np.linspace(0, in_of_samples / 4, in_of_samples)
    iy = adcl["Res_Espinheira"][:in_of_samples * (1 if is_resampled else 15)]

    yi = []

    if not len(iy) < iwindow:
        for i in range(len(iy) - iwindow):
            yi.append(np.mean([iy[j + i] for j in range(iwindow)]))
    if not is_resampled:
        yi = resample(adcl, '15min')["Res_Espinheira"][:in_of_samples]
    plt.plot(ix[:len(yi)], yi, color=icolor, label=my_label)
    plt.legend(loc='upper right')


def rolling(x, y):
    time = 15
    yi = y.copy().rolling(window=15)
    yi = yi.mean()
    # yi = my_resample(yi, time)
    plt.plot(x[:len(x)], yi, color='sienna')


def original(x, y, icolor):
    plt.plot(x, y, color=icolor, label='Original')
    plt.legend(loc='upper right')


def smoothness_array(arr):
    arr2 = []
    for i in range(len(arr) - 2):
        arr2.append(smoothness_level(arr[i], arr[i + 1], arr[i + 2]))

    return arr2


def smoothness_array_dif(arr):
    result = []
    for i in range(len(arr) - 2):
        result.append((arr[i + 2] - arr[i + 1]) - (arr[i + 1] - arr[i]))

    return np.mean(result)


def smoothness_level(y1, y2, y3):
    angle = f1(y1, y2, y3)

    return angle


def f1(x1, x2, x3):
    numerator = 1 + (x3 - x2) * (x2 - x1)
    denominator = (1 + (x3 - x2) ** 2) * (1 + (x2 - x1) ** 2)
    denominator = np.sqrt(denominator)

    return np.rad2deg(np.arccos(numerator / denominator))


def f2(ang):
    return 100 - np.log(ang / 180 + 1) / np.log(2)


def my_print(array_of_tuples):
    for elem in array_of_tuples:
        print(str(elem[1]) + ": " + str(elem[0]))


if __name__ == '__main__':
    # font_network = Fontinha()
    # sim_data = font_network.generate_sim_data(n_batches=2000)
    # sim_data.to_csv('simDataFontinha.csv', index=False)

    # rich_network = Richmond(sim_step=1800)
    # sim_data = rich_network.generate_sim_data(n_batches=2000)
    # sim_data.to_csv('simDataRichmond30min.csv', index=False)

    adcl_raw = pd.read_csv("adcl_data.csv")
    adcl_raw["Time"] = pd.to_datetime(adcl_raw['Time'], utc=True)
    adcl_raw.set_index('Time', inplace=True)
    adcl_raw.dropna(inplace=True)

    adcl_valve = calc_valves_from_tanks_adcl(adcl_raw)
    adcl_valve.to_csv("adcl_data_valve.csv", index=True)
    periodicities = ['15min', '1h']
    for per in periodicities:
        adcl_valve_per = resample(adcl_raw, periodicity=per)
        adcl_valve_per = calc_valves_from_tanks_adcl(adcl_valve_per)
        adcl_valve_per.dropna(inplace=True)

        adcl_valve_per.to_csv(f"adcl_data_valve_{per}.csv", index=True)


