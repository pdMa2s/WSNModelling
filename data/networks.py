import random
import pandas as pd
import numpy as np

from epanet.epamodule import ENopen, ENopenH, ENinitH, ENsettimeparam, EN_DURATION, ENrunH, ENgetnodevalue, ENgetcount, \
    EN_NODECOUNT, ENgetnodetype, EN_TANK, EN_PRESSURE, ENnextH, EN_LINKCOUNT, ENgetlinktype, EN_PUMP, ENgetlinkvalue, \
    EN_ENERGY, EN_JUNCTION, EN_DEMAND, EN_STATUS, ENsetnodevalue, ENsetlinkvalue, EN_TANKLEVEL, EN_MINLEVEL, \
    EN_MAXLEVEL, EN_ELEVATION, ENcloseH, ENclose, EN_HYDSTEP, ENgetnodeid
from data.benchmark_solution import benchmark2018, get_demand
from numpy import array


def calculate_aggregated_demand(time_incs):
    for i in range(0, len(time_incs)):
        dmd_sum = sum(time_incs[i]['dmds'])
        if i > 0:
            dmd_sum += time_incs[i-1]['agrg']
        time_incs[i]['agrg'] = dmd_sum


def create_dataset_columns(time_inc, keys):
    columns = []
    for k in keys:
        value = time_inc[k]
        if isinstance(value, list):
            for i in range(0, len(time_inc[k])):
                columns.append(k[:3] + str(i))
        else:
            columns.append(k)
    return columns


# process the output of the benchmark and returns a data frame
# the demand columns are added at the end of the dataset columns
def output_to_dataset(time_incs):
    time_keys = ['hFfin', 'E', 'hFini', 'dmds', 'agrg', 'pumps']
    dataset_columns = create_dataset_columns(time_incs[0], time_keys)
    dataset = pd.DataFrame(columns=dataset_columns)
    for i in range(0, len(time_incs)):
        row_values = list()
        for k in range(len(time_keys)):
            if isinstance(time_incs[i][time_keys[k]], list):
                for d in time_incs[i][time_keys[k]]:
                    row_values.append(d)
            else:
                row_values.append(time_incs[i][time_keys[k]])

        row = pd.DataFrame([row_values], columns=dataset_columns)
        dataset = dataset.append(row)
    return dataset


class Network:
    def generate_sim_data(self, demand_data=None, n_batches=10):
        pass


class Fontinha(Network):
    def __get_final_tank_h__(self, time_inc: dict):
        return time_inc[23]['hFfin']

    def generate_sim_data(self, demand_data=None, n_batches=10):
        time_inc_list = list()

        batch = 24
        h_initial = random.randint(2, 7)
        for i in range(0, n_batches):

            if h_initial <= 2:
                pump_settings = [1, 1] + [round(random.uniform(0, 1), 3) for _ in range(0, batch - 2)]
            elif h_initial >= 7:
                pump_settings = [0, 0] + [round(random.uniform(0, 1), 3) for _ in range(0, batch - 2)]
            else:
                pump_settings = [round(random.uniform(0, 1), 3) for _ in range(0, batch)]

            f_obj_rest, sensibil, time_inc = benchmark2018(pump_settings, 0, hF0=h_initial)

            h_initial = self.__get_final_tank_h__(time_inc)
            # calculate the agregated demand per batch and add a new key to the time increment
            calculate_aggregated_demand(time_inc)
            time_inc_list.append(time_inc)

        time_inc_list = array(time_inc_list).flatten()

        dataset = output_to_dataset(time_inc_list)
        dataset.info()

        return dataset


class Richmond(Network):
    def __init__(self, network_file='../epanet/Richmond_skeleton.inp', sim_duration=24, sim_step=3600, hydraulic_step=10):
        self.sim_duration = sim_duration * 3600
        self.empty_time_inc = {
            'startTime': None, 'duration': None, 'endTime': None,
            'hFini': [], 'hFfin': [], 'E': -1, 'dmds': [], 'pumps': []}

        #ENopen('../epanet/Richmond_skeleton.inp', '/dev/null')
        ENopen(network_file, '/dev/null')

        ENsettimeparam(EN_DURATION, self.sim_duration)
        ENsettimeparam(EN_HYDSTEP, hydraulic_step)
        self.sim_step = sim_step
        self.tank_idxs = self.node_indexes(EN_TANK)
        self.pump_idxs = self.link_indexes(EN_PUMP)
        self.junction_idxs = self.node_indexes(EN_JUNCTION)

    def __initialize_tank_levels__(self, tank_idx_list):
        pass

    # returns the indexes of elements that have a given code
    # this method can not be static
    def _get_indexes(self, count_code, type_func, type_code):
        n_links = ENgetcount(count_code)
        indexes = []
        for i in range(1, n_links + 1):
            type = type_func(i)
            if type == type_code:
                indexes.append(i)
                # print(f"Index {i}: {ENgetnodeid(i)}")
        return indexes

    def node_indexes(self, node_code):
        return self._get_indexes(EN_NODECOUNT, ENgetnodetype, node_code)

    def link_indexes(self, link_code):
        return self._get_indexes(EN_LINKCOUNT, ENgetlinktype, link_code)

    def _get_values(self, indexes, value_code, get_func=ENgetnodevalue):
        values = []
        for i in indexes:
            values.append(get_func(i, value_code))
        return values

    def _set_values(self, indexes, property_code, values, set_func=ENsetnodevalue):
        for i in range(len(indexes)):
            set_func(indexes[i], property_code, values[i])

    def get_tank_levels(self, indexes):
        return self._get_values(indexes, EN_PRESSURE, get_func=ENgetnodevalue)

    def get_junction_demands(self, indexes):
        return self._get_values(indexes, EN_DEMAND, get_func=ENgetnodevalue)

    def get_pump_statuses(self, indexes):
        return self._get_values(indexes, EN_STATUS, get_func=ENgetlinkvalue)

    def calculate_pump_times(self, previous_statuses, time_step):
        times = []
        for i in range(len(previous_statuses)):
            if previous_statuses[i] == 1:
                times.append(time_step)
            else:
                times.append(0)
        return times

    def get_pump_energy(self, pump_indexes, step):
        if step < 0:
            return 0
        return sum(self._get_values(pump_indexes, EN_ENERGY, ENgetlinkvalue)) * (step / 3600)

    def _set_pumps(self, indexes, settings, iteration):
        if settings is not None and iteration < self.sim_duration:
            self._set_values(indexes, EN_STATUS, settings[iteration], set_func=ENsetlinkvalue)

    def _set_tanks_levels(self, indexes, levels):
        if levels is not None:
            self._set_values(indexes, EN_TANKLEVEL, levels, set_func=ENsetnodevalue)

    def _set_demands(self, indexes, demands, iteration):
        if demands is not None and iteration < self.sim_duration:
            self._set_values(indexes, EN_DEMAND, demands[iteration], set_func=ENsetnodevalue)

    def __run_simulation__(self, pump_settings=None, tanks_initial_levels=None, demands=None):
        it = 0
        hydraulic_t_step = -1
        energy_sum = 0

        time_inc_list = []
        self._set_tanks_levels(self.tank_idxs, tanks_initial_levels)
        pump_time_list = [0 for _ in range(len(self.pump_idxs))]
        previous_pump_statuses = []
        while hydraulic_t_step != 0:

            t = ENrunH()

            energy_sum += self.get_pump_energy(self.pump_idxs, hydraulic_t_step)
            if t == 0:
                previous_pump_statuses = self.get_pump_statuses(self.pump_idxs)
            else:
                inc_pump_time = self.calculate_pump_times(previous_pump_statuses, hydraulic_t_step)
                previous_pump_statuses = self.get_pump_statuses(self.pump_idxs)
                pump_time_list = [sum(x) for x in zip(pump_time_list, inc_pump_time)]

            if t % self.sim_step is 0:
                self._set_pumps(self.pump_idxs, pump_settings, it)
                self._set_demands(self.junction_idxs, demands, it)

                time_inc = self.empty_time_inc.copy()
                time_inc['hFini'] = self.get_tank_levels(self.tank_idxs)
                time_inc['dmds'] = self.get_junction_demands(self.junction_idxs)

                if it > 0:
                    time_inc_list[it - 1]['hFfin'] = time_inc['hFini']
                    time_inc_list[it - 1]['E'] = energy_sum
                    time_inc_list[it - 1]['pumps'] = [time / self.sim_step for time in pump_time_list]
                    time_inc_list[it - 1]['dmds'] = self.get_junction_demands(self.junction_idxs)

                    energy_sum = 0
                    pump_time_list = [0 for _ in range(len(self.pump_idxs))]

                time_inc_list.append(time_inc)
                it += 1

            hydraulic_t_step = ENnextH()
        del time_inc_list[-1]

        return time_inc_list

    # some random values generate errors
    def __change_tank_limit_levels__(self, indexes):
        self._set_values(indexes, EN_MAXLEVEL, [random.randint(4, 20) for _ in range(len(indexes))],
                         set_func=ENsetnodevalue)

        self._set_values(indexes, EN_MINLEVEL, [random.randint(0, ENgetnodevalue(self.tank_idxs[i], EN_MAXLEVEL))
                                                for i in range(len(indexes))], set_func=ENsetnodevalue)

    def generate_sim_data(self, demand_data=None, n_batches=10):
        time_inc_list = list()
        for b in range(n_batches):
            ENopenH()

            ENinitH(10)

            initial_levels = [random.uniform(ENgetnodevalue(self.tank_idxs[i], EN_MINLEVEL),
                                             ENgetnodevalue(self.tank_idxs[i], EN_MAXLEVEL))
                              for i in range(len(self.tank_idxs))]

            pump_statuses = [[random.randint(0, 1) for _ in range(len(self.pump_idxs))]
                             for _ in range(int(self.sim_duration / self.sim_step)+1)]

            demands = [[random.uniform(0, 3) if x != len(self.junction_idxs) - 1 else random.uniform(-10, 0)
                        for x in range(len(self.junction_idxs))]
                       for _ in range(int(self.sim_duration/self.sim_step)+1)]

            time_incs = self.__run_simulation__(tanks_initial_levels=initial_levels,
                                                pump_settings=None, demands=demands)
            calculate_aggregated_demand(time_incs)

            time_inc_list.extend(time_incs)

            ENcloseH()

        dataset = output_to_dataset(time_inc_list)
        dataset.info()
        return dataset


if __name__ == '__main__':
    # rich = Richmond()
    # time_incs = rich.generate_sim_data()
    font = Fontinha()
    time_incs = font.generate_sim_data()
    print(time_incs)