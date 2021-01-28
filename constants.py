import pytz

TOTAL_SECONDS_IN_HOUR = 3600
TOTAL_HOURS_IN_DAY = 24

#  _________        _        _______      _____   ________   ________    ______
# |  _   _  |      / \      |_   __ \    |_   _| |_   __  | |_   __  | .' ____ \
# |_/ | | \_|     / _ \       | |__) |     | |     | |_ \_|   | |_ \_| | (___ \_|
#     | |        / ___ \      |  __ /      | |     |  _|      |  _|     _.____`.
#    _| |_     _/ /   \ \_   _| |  \ \_   _| |_   _| |_      _| |_     | \____) |
#   |_____|   |____| |____| |____| |___| |_____| |_____|    |_____|     \______.'

PEAK_SUMMER = 0.2282
HALFPEAK_SUMMER = 0.1132
OFFPEAK_SUMMER = 0.0876
SUPEROFFPEAK_SUMMER = 0.0797

PEAK_WINTER = 0.1845
HALFPEAK_WINTER = 0.1064
OFFPEAK_WINTER = 0.0813
SUPEROFFPEAK_WINTER = 0.0739

#   ______    _____   ____    ____   _____  _____    _____           _        _________   _____     ___     ____  _____
# .' ____ \  |_   _| |_   \  /   _| |_   _||_   _|  |_   _|         / \      |  _   _  | |_   _|  .'   `.  |_   \|_   _|
# | (___ \_|   | |     |   \/   |     | |    | |      | |          / _ \     |_/ | | \_|   | |   /  .-.  \   |   \ | |
#  _.____`.    | |     | |\  /| |     | '    ' |      | |   _     / ___ \        | |       | |   | |   | |   | |\ \| |
# | \____) |  _| |_   _| |_\/_| |_     \ \__/ /      _| |__/ |  _/ /   \ \_     _| |_     _| |_  \  `-'  /  _| |_\   |_
#  \______.' |_____| |_____||_____|     `.__.'      |________| |____| |____|   |_____|   |_____|  `.___.'  |_____|\____|

LISBON = pytz.timezone("Europe/Lisbon")

CORRESPONDING_TANK_DICT = {
            b'P_Aveleira': 'Res_Aveleira',
            b'P_Albarqueira': 'Res_Albarqueira_cel1',
            b'ETARQ_Q2': 'Res_Travanca',
            b'ETARQ_Q3': 'Res_OCastro',
            b'ETARQ_Q4': 'Res_SPDias'
}

DEFAULT_CONTROL = "LINK Valve1 1.0000 AT TIME 0.0000 HOURS\n"

PUMPS_NOMINAL_POWERS = {
            'B4N': 162,
            'B8N': 61,
            'B5': 51.8,
            'B9': 16,
            'B10': 15.4,
            'B9+B10': 28.5
}  # As read from Fase1-Implementação SCUBIC Fervenca.docs

EPANET_MAX_TRIALS = 5000
EPANET_DEFAULT_BASEDEMAND = 1
READING_FREQ_TANKS_WITH_NO_CONTROLS = 1800

#    ___     _______    _________   _____   ____    ____   _____   ________        _        _________   _____     ___     ____  _____
#  .'   `.  |_   __ \  |  _   _  | |_   _| |_   \  /   _| |_   _| |  __   _|      / \      |  _   _  | |_   _|  .'   `.  |_   \|_   _|
# /  .-.  \   | |__) | |_/ | | \_|   | |     |   \/   |     | |   |_/  / /       / _ \     |_/ | | \_|   | |   /  .-.  \   |   \ | |
# | |   | |   |  ___/      | |       | |     | |\  /| |     | |      .'.' _     / ___ \        | |       | |   | |   | |   | |\ \| |
# \  `-'  /  _| |_        _| |_     _| |_   _| |_\/_| |_   _| |_   _/ /__/ |  _/ /   \ \_     _| |_     _| |_  \  `-'  /  _| |_\   |_
#  `.___.'  |_____|      |_____|   |_____| |_____||_____| |_____| |________| |____| |____|   |_____|   |_____|  `.___.'  |_____|\____|


# When filtering orders with less than x minutes, if cost raises by MAX_WORSENING_THRESHOLD, pass a new filter that
# eliminates orders with less than BACKUP_FILTER
BACKUP_FILTER = "2Min"

# When filtering orders with less than x minutes, if cost raises by MAX_WORSENING_THRESHOLD, filtering is revoked
MAX_WORSENING_THRESHOLD = 1.05

SEED_VALUE = 0.5
SEED_METHOD = "last_x"
MIN_DURATION_OPTIMIZATION_VARIABLE = '4H'

OPTIMIZED_TANKS = ['Res_Espinheira', 'Res_Aveleira', 'Res_Albarqueira_cel1', 'Res_Travanca', 'Res_OCastro', 'Res_SPDias']
OPTIMIZED_DEMANDS = ['PE_Espinheira', 'PE_Aveleira', 'PE_Albarqueira']

PUMP_TRANSLATION = {b'RAVL_Q1': 'Granja', b'ETARQ_Q1': 'Albarqueira', b'ETARQ_Q2': 'Travanca', b'ETARQ_Q3': 'O. de Castro',
                    b'ETARQ_Q4': 'S. Pedro Dias'}
OPTIMIZED_PUMPS = PUMP_TRANSLATION.keys()

#  _____   ____  _____   ________    _____      _____  _____   ____  ____             _________        _          ______     ______
# |_   _| |_   \|_   _| |_   __  |  |_   _|    |_   _||_   _| |_  _||_  _|           |  _   _  |      / \       .' ___  |  .' ____ \
#   | |     |   \ | |     | |_ \_|    | |        | |    | |     \ \  / /             |_/ | | \_|     / _ \     / .'   \_|  | (___ \_|
#   | |     | |\ \| |     |  _|       | |   _    | '    ' |      > `' <                  | |        / ___ \    | |   ____   _.____`.
#  _| |_   _| |_\   |_   _| |_       _| |__/ |    \ \__/ /     _/ /'`\ \_               _| |_     _/ /   \ \_  \ `.___]  | | \____) |
# |_____| |_____|\____| |_____|     |________|     `.__.'     |____||____|             |_____|   |____| |____|  `._____.'   \______.'

OPERATION_TYPE_TAG = 'operational_type'
OPERATION_TYPE_PREDICT = 'predicted'
OPERATION_TYPE_REAL = 'real'
PUMP_TAG = 'pumpID'
SYSTEM_ID = "system"
DEMAND_NODE_TAG = 'node_id'
KPI_TAG = 'KPI'
KPI_SPECIFIC_OPERATION_COST = 'specific_operation_cost'
KPI_SPECIFIC_ENERGY_COST = 'specific_energy_cost'
KPI_ORDERS_FULFILLMENT_RATIO = 'ratio' # TODO: the name is not very intuitive
KPI_ORDERD_FULFILLMENT_DEVIATION = 'deviation' # TODO: the name is not very intuitive
KPI_COST = 'cost'
KPI_VOLUME = 'volume'
KPI_ENERGY = 'energy'
KPI_NASH_SUTCLIFFE = 'nse'
KPI_RMSE = 'rmse'
KPI_MEAN_ERROR = 'me'
KPI_MEAN_ABSOLUTE_ERROR = 'mae'
KPI_ENERGY_DISTRIBUTION_PER_TARIFF = 'energy_distribution'
KPI_TARIFF_USAGE_EFFICIENCY = 'tariff_usage_efficiency'
KPI_RATIO_TO_OPTIMUM = 'optimum_ratio'

PUMP_STATE_FIELD = "state"
PUMP_STATE_MEASUREMENT = 'pump_state'
TANK_LEVEL_MEASUREMENT = 'tank_level'
TANK_LEVEL_FORECAST_MEASUREMENT = 'tank_level_forecast'

TARIFF_PERIOD_POSSIBLE_VALUES = ['peak', 'halfpeak', 'offpeak', 'super_offpeak']
TARIFF_PERIOD_TAG = 'tariff'

FORECAST_MODEL_TAG = 'model'
BEST_FORECAST_MODEL_TAG_VALUE = 'Best'


#  ________     ___     _______      ________     ______        _         ______    _________    ______
# |_   __  |  .'   `.  |_   __ \    |_   __  |  .' ___  |      / \      .' ____ \  |  _   _  | .' ____ \
#   | |_ \_| /  .-.  \   | |__) |     | |_ \_| / .'   \_|     / _ \     | (___ \_| |_/ | | \_| | (___ \_|
#   |  _|    | |   | |   |  __ /      |  _| _  | |           / ___ \     _.____`.      | |      _.____`.
#  _| |_     \  `-'  /  _| |  \ \_   _| |__/ | \ `.___.'\  _/ /   \ \_  | \____) |    _| |_    | \____) |
# |_____|     `.___.'  |____| |___| |________|  `.____ .' |____| |____|  \______.'   |_____|    \______.'

OBSERVATION_WINDOW = 100

DROPNAN = True
IS_PREDICT = True
OUTLIER_THRESHOLD = 3
N_FEATURES = 14
FORECAST_WINDOW = "24h"
PERIODICITY = "15Min"
MODELS_PATH = "models/"
