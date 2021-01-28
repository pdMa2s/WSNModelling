import pandas as pd
from matplotlib import pyplot as plt
import numpy


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

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


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

