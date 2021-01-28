import datetime
import pandas as pd
import holidays
import DBconnectors.constants as constants
from DBconnectors.utils import utc_to_local

PEAK_SUMMER = constants.PEAK_SUMMER
HALFPEAK_SUMMER = constants.HALFPEAK_SUMMER
OFFPEAK_SUMMER = constants.OFFPEAK_SUMMER
SUPEROFFPEAK_SUMMER = constants.SUPEROFFPEAK_SUMMER

PEAK_WINTER = constants.PEAK_WINTER
HALFPEAK_WINTER = constants.HALFPEAK_WINTER
OFFPEAK_WINTER = constants.OFFPEAK_WINTER
SUPEROFFPEAK_WINTER = constants.SUPEROFFPEAK_WINTER

MIDDAY = datetime.time(hour=12)


def localize_lisbon_date(date):
    return pd.to_datetime(date).tz_localize('Europe/Lisbon')


def is_dst(_pd_datetime):
    """
    Determines of a given datetime is daylight saving (Summer) or not (Winter)

    :param _pd_datetime:
    :type _pd_datetime: pandas.Timestamp
    :return: True if _pd_datetime is dst else False
    :rtype: bool
    """
    dst_diff = _pd_datetime.tz._dst
    if dst_diff > datetime.timedelta(0):
        return True
    else:
        return False


def get_super_empty_tariff(date):
    return SUPEROFFPEAK_SUMMER if is_dst(date) else SUPEROFFPEAK_WINTER


def get_tariff(start_day: datetime.datetime, end_day: datetime.datetime) -> (list, list, list):
    """
    Return starts, ends and tariffs price for a optimization window

    Starts represent the start instant (in seconds) for each optimization period
    Ends represent the end instant (in seconds) for each optimization period
    Tariffs represent the price (in euros) for each optimization period

    :param start_day: date of the start instant of optimization
    :type start_day: datetime.datetime
    :param end_day: date of the end instant of optimization
    :type end_day: datetime.datetime
    :return: (Starts, Ends, Tariffs)
    :rtype: (list, list, list)
    """

    start_day = pd.to_datetime(start_day).tz_localize('Europe/Lisbon')
    end_day = pd.to_datetime(end_day).tz_localize('Europe/Lisbon')
    tariffs_df = pd.DataFrame()
    tariffs_df['times'] = pd.date_range(start_day, end_day, freq='15Min')
    tariffs_df['tar'] = tariffs_df['times'].apply(calc_price)
    tariffs_df['flag'] = (tariffs_df['tar'].shift() == tariffs_df['tar'])
    tariffs_df = pd.concat([tariffs_df.loc[tariffs_df['flag'] == 0], pd.DataFrame(tariffs_df.tail(1))])
    tariffs_df.drop_duplicates(inplace=True)
    tariffs_df['flag'] = (tariffs_df['times'].diff() > pd.Timedelta(constants.MIN_DURATION_OPTIMIZATION_VARIABLE))
    while sum(tariffs_df['flag'] > 0):
        mask1 = tariffs_df.where(tariffs_df['flag'].shift(-1) == 1).dropna().reset_index()
        mask2 = tariffs_df.where(tariffs_df['flag'] == 1).dropna().reset_index()
        result = pd.DataFrame()
        result['times'] = mask1['times'] + (mask2['times'] - mask1['times']) / 2
        result['tar'] = mask1['tar']
        result['index'] = (mask2['index'] + mask1['index']) / 2
        result.set_index('index', inplace=True)
        tariffs_df = tariffs_df.append(result).sort_index()
        tariffs_df.drop('flag', axis=1, inplace=True)
        tariffs_df['flag'] = (tariffs_df['times'].diff() > pd.Timedelta(constants.MIN_DURATION_OPTIMIZATION_VARIABLE))
    tariffs_df.reset_index(inplace=True)
    tariffs_df['duration'] = tariffs_df['times'].diff().shift(-1).fillna(pd.Timedelta('0H'))
    tariffs_df['ends'] = tariffs_df['duration'].dt.total_seconds().cumsum().astype(int)
    tariffs_df['starts'] = tariffs_df['ends'].shift(1).fillna(0)
    tariffs_df = tariffs_df.where(tariffs_df['duration'] != pd.Timedelta('0S')).dropna().reset_index()

    starts = tariffs_df['starts'].to_list()
    ends = tariffs_df['ends'].to_list()
    tars = tariffs_df['tar'].to_list()
    return starts, ends, tars


def is_date_naive(date):
    return date.tzinfo is None or date.tzinfo.utcoffset(date) is None


def calc_price(date):

    holidays_list = holidays.PT()

    if is_date_naive(date):
        date = localize_lisbon_date(date)
    else:
        date = utc_to_local(date)

    date_is_dst = is_dst(date)
    date_midnight = pd.Timestamp(date.year, date.month, date.day).tz_localize('Europe/Lisbon')
    if date_is_dst:  # Summer
        p = Peak(_is_dst=date_is_dst)
        f = HalfPeak(_is_dst=date_is_dst)
        e = OffPeak(_is_dst=date_is_dst)
        s = SuperOffPeak(_is_dst=date_is_dst)
        if date.weekday() in [0, 1, 2, 3, 4]:  # Weekday
            if date_midnight + pd.Timedelta('6H') > date >= date_midnight + pd.Timedelta('2H'):
                return s
            elif date_midnight + pd.Timedelta('2H') > date >= date_midnight + pd.Timedelta('0H') or \
                    date_midnight + pd.Timedelta('7H') > date >= date_midnight + pd.Timedelta('6H'):
                return e
            elif date_midnight + pd.Timedelta('9H15Min') > date >= date_midnight + pd.Timedelta('7H') or \
                    date_midnight + pd.Timedelta('24H') > date >= date_midnight + pd.Timedelta('12H15Min'):
                return f
            elif date_midnight + pd.Timedelta('12H15Min') > date >= date_midnight + pd.Timedelta('9H15Min'):
                return p

        elif date.weekday() == 5:  # Saturday
            if date_midnight + pd.Timedelta('6H') > date >= date_midnight + pd.Timedelta('2H'):
                return s
            elif date_midnight + pd.Timedelta('14H') > date >= date_midnight + pd.Timedelta('9H') or \
                    date_midnight + pd.Timedelta('22H') > date >= date_midnight + pd.Timedelta('20H'):
                return f
            else:
                return e

        if date.weekday() == 6 or date in holidays_list:  # Sunday
            if date_midnight + pd.Timedelta('6H') > date >= date_midnight + pd.Timedelta('2H'):
                return s
            else:
                return e
    elif not date_is_dst:  # Winter
        p = Peak(_is_dst=date_is_dst)
        f = HalfPeak(_is_dst=date_is_dst)
        e = OffPeak(_is_dst=date_is_dst)
        s = SuperOffPeak(_is_dst=date_is_dst)
        if date.weekday() in [0, 1, 2, 3, 4]:  # Weekday
            if date_midnight + pd.Timedelta('6H') > date >= date_midnight + pd.Timedelta('2H'):
                return s
            elif date_midnight + pd.Timedelta('2H') > date >= date_midnight + pd.Timedelta('0H') or \
                    date_midnight + pd.Timedelta('7H') > date >= date_midnight + pd.Timedelta('6H'):
                return e
            elif date_midnight + pd.Timedelta('9H30Min') > date >= date_midnight + pd.Timedelta('7H') or \
                    date_midnight + pd.Timedelta('18H30Min') > date >= date_midnight + pd.Timedelta('12H') or \
                    date_midnight + pd.Timedelta('24H') > date >= date_midnight + pd.Timedelta('21H'):
                return f
            else:
                return p

        elif date.weekday() == 5:  # Saturday
            if date_midnight + pd.Timedelta('6H') > date >= date_midnight + pd.Timedelta('2H'):
                return s
            elif date_midnight + pd.Timedelta('13H') > date >= date_midnight + pd.Timedelta('9H30Min') or \
                    date_midnight + pd.Timedelta('22H') > date >= date_midnight + pd.Timedelta('18H30Min'):
                return f
            else:
                return e
        if date.weekday() == 6 or date in holidays_list:  # Sunday
            if date_midnight + pd.Timedelta('6H') > date >= date_midnight + pd.Timedelta('2H'):
                return s
            else:
                return e


class Tariff:
    summer_price: float = -1
    winter_price: float = -1
    name: str = "tariff"
    is_dst: bool = False
    value: float = -1

    def __init__(self, summer_price: float = -1, winter_price: float = -1, _is_dst: bool = -1, name: str = "tariff",
                 value: float = -1):
        self.summer_price = summer_price
        self.winter_price = winter_price
        self.name = name
        self.is_dst = _is_dst
        self.value = value

    def get_price(self):
        return self.value if self.value != -1 else self.summer_price if self.is_dst else self.winter_price

    def __str__(self):
        return f"{self.name} = {self.get_price()}"

    def __repr__(self):
        return f"Tariff(summer_price={self.summer_price}, winter_price={self.winter_price}, _is_dst={self.is_dst}, " \
               f"name={self.name})"

    def __rmul__(self, other):
        return self.get_price() * other

    def __mul__(self, other):
        return self.get_price() * other

    def __ge__(self, other):
        return self.get_price() >= other

    def __le__(self, other):
        return self.get_price() <= other

    def __gt__(self, other):
        return self.get_price() > other

    def __lt__(self, other):
        return self.get_price() < other

    def __eq__(self, other):
        return self.name == other.name

    def __add__(self, other):
        return self.get_price() + other

    def __sub__(self, other):
        return self.get_price() - other

    def __ne__(self, other):
        return self.get_price() != other

    def __hash__(self) -> int:
        return hash(self.name)


class OffPeak(Tariff):
    def __init__(self, summer_price: float = OFFPEAK_SUMMER, winter_price: float = OFFPEAK_WINTER,
                 _is_dst: bool = False, value: float = -1):
        super().__init__(summer_price, winter_price, _is_dst, "off peak", value)


class SuperOffPeak(Tariff):
    def __init__(self, summer_price: float = SUPEROFFPEAK_SUMMER, winter_price: float = SUPEROFFPEAK_WINTER,
                 _is_dst: bool = False, value: float = -1):
        super().__init__(summer_price, winter_price, _is_dst, "super off peak", value)


class HalfPeak(Tariff):
    def __init__(self, summer_price: float = HALFPEAK_SUMMER, winter_price: float = HALFPEAK_WINTER,
                 _is_dst: bool = False, value: float = -1):
        super().__init__(summer_price, winter_price, _is_dst, "half peak", value)


class Peak(Tariff):
    def __init__(self, summer_price: float = PEAK_SUMMER, winter_price: float = PEAK_WINTER,
                 _is_dst: bool = False, value: float = -1):
        super().__init__(summer_price, winter_price, _is_dst, "peak", value)


if __name__ == '__main__':
    date_summer = '2019-08-01 05:23:44'
    tup_summer = get_tariff(date_summer)

    date_winter = '2019-02-01 05:23:44'
    tup_winter = get_tariff(date_winter)

    date_spring1 = '2019-03-30 05:23:44'
    tup_spring1 = get_tariff(date_spring1)

    date_spring2 = '2019-03-31 05:23:44'
    tup_spring2 = get_tariff(date_spring2)

    date_fall1 = '2019-10-26 05:23:44'
    tup_fall1= get_tariff(date_fall1)

    date_fall2 = '2019-10-27 05:23:44'
    tup_fall2 = get_tariff(date_fall2)

    date_christmas = '2019-12-25 05:23:44'
    tup_christmas = get_tariff(date_christmas)

    print(tup_summer)
    print(tup_winter)
    print(tup_spring1)
    print(tup_spring2)
    print(tup_fall1)
    print(tup_fall2)
    print(tup_christmas)

    print(' ')
