from typing import List, Dict
import os
import tempfile
import logging
from functools import lru_cache
import datetime
import pytz
import numpy as np

from jinja2 import Template
import pandas as pd
from epanet import epamodule
from typing import Union
import constants
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from data_utils import nash_sutcliffe, plot_results_lines, load_adcl_raw, load_adcl

logger = logging.getLogger()
LISBON = pytz.timezone("Europe/Lisbon")


class SimulationResults:
    """
    Object used to return results from a complete simulation
    """
    tank_levels: Dict[str, List[float]]
    tank_times: Dict[str, List[float]]
    tank_min_level: Dict[str, List[float]]
    tank_max_level: Dict[str, List[float]]
    cost: float
    pumps: List
    energy: float

    def __init__(self, tanks, cost, pumps):
        """

        :param tanks: Array of @Tank objects
        :param power: Dictionary with the power values of the pumps
        :param cost: The total cost of this simulation
        :param pumps: Array of @Pump objects
        """
        self.tank_levels = {}
        self.tank_times = {}
        self.tank_min_level = {}
        self.tank_max_level = {}

        for tank_id, tank in tanks.items():
            self.tank_levels[tank_id] = tank.simulation_levels
            self.tank_times[tank_id] = tank.simulation_times
            # self.tank_min_level[tank_id] = tank.min_level
            # self.tank_max_level[tank_id] = tank.max_level

        self.cost = cost
        self.pumps = pumps
        self.tanks = tanks

    def levels(self, simulator) -> list:
        """
        The levels of each tank
        :return:
        """
        simulation_levels = []
        n_considered_tanks = 0
        for tank_id, tank in simulator.tanks.items():
            for pump in simulator.pumps:
                corresponding_tank = pump.get_corresponding_tank()
                if tank_id == corresponding_tank:
                    # print(tank.simulation_levels)
                    simulation_levels += tank.simulation_levels
                    n_considered_tanks += 1
                    break

        # print(simulation_levels)
        return simulation_levels
        # return sum([self.tank_levels[l] for l in self.tank_levels], [])

    @property
    def min_levels(self) -> List[float]:
        """
        :return: List with min admissible water level for each tank
        """
        return [min(self.tank_levels[l]) for l in self.tank_levels]

    @property
    def max_levels(self) -> List[float]:
        """
        :return: List with max admissible water level for each tank
        """
        return [max(self.tank_levels[l]) for l in self.tank_levels]

    def get_pump_times(self, start):
        """
        Process a dictionary with the times where each pump is turned on and off
        :param start: A datetime that marks the beginning of the simulation
        :return: A dict that contains the datetime objects for each start and shutdown of each pump
        """
        pumps_dict = {}
        for pump in self.pumps:
            dataframe_ = pd.DataFrame()
            time = []
            command = []
            for i in range(len(pump.start_intervals)):
                t_on = pump.start_intervals[i].epanet_on_time
                t_off = pump.start_intervals[i].epanet_off_time
                time += [start + t_on * pd.Timedelta("1S"),
                         start + t_off * pd.Timedelta("1S")]
                command += [1, 0]
            dataframe_['Time'] = time
            dataframe_[pump.link_id] = command
            pumps_dict[pump.link_id] = dataframe_
        return pumps_dict

    def get_tank_levels(self, start):
        """
        Get a dictionary with every tank where the value is a DataFrame \
        with the tank levels in the time interval of the simulation
        :param start: A datetime that marks the beginning of the simulation
        :return: A dict that contains the tank level during the simulation
        """
        tanks_dict = {}
        for tank in self.tank_levels:
            dataframe_ = pd.DataFrame()
            dataframe_['Time'] = list(map(lambda x: start + x * pd.Timedelta('1S'), self.tank_times[tank]))
            dataframe_.tail(1)['Time'] -= pd.Timedelta('1S')
            dataframe_[tank] = self.tank_levels[tank]
            tanks_dict[tank] = dataframe_
        return tanks_dict


class ControlOperation:
    def __init__(self, time: int, operation: Union[int, float]):
        self.time = time
        self.operation = operation
        self.epanet_time = None

    def __repr__(self):
        return f"time: {self.time} | op: {self.operation} | epanet_time: {self.epanet_time}"


class EpanetElement:
    def __init__(self, en_id: Union[str, bytes], en_index: int):
        assert en_id is not None and en_index is not None
        assert isinstance(en_id, (str, bytes)) and isinstance(en_index, int)

        self.en_id = en_id
        self.en_index = en_index
        self.controls = []

    def add_control_operation(self, op_time, operation):
        self.controls.append(ControlOperation(op_time, operation))

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.en_id} | EN_index: {self.en_index}"


class SimulationPump(EpanetElement):
    """
    Class representing a Pump in the simulation

    :param link_id: The id of pump that is given by the epanet file
    :type link_id: bytes
    :param en_index: The index of the pump on epanet
    :type en_index: int
    :param start: The of the optimization window
    :type start: datetime.datetime
    :param speeds: The speeds of the VSD of the pump for its respective optimization periods. This is given by a real
    number that can take values of {0, [0.6, 1.2]}
    :type speeds: List[float]
    """

    link_id: bytes
    en_index: int
    cost: float = 0
    energy: float = 0
    clocktime: List = []
    powers_reads: List = []
    flow_reads = List = []

    def __init__(self, link_id: bytes, en_index: int, on_off_times: Union[list, None] = None):
        super().__init__(link_id, en_index)
        assert isinstance(on_off_times, list) or on_off_times is None

        self.controls = [ControlOperation(op[0], op[1]) for op in on_off_times] if on_off_times is not None else []

    def __str__(self):
        string_representation = f'PumpID: {self.link_id} | EN_index: {self.en_index} |' \
                                f' Corresponding Tank: {self.get_corresponding_tank()}'
        return string_representation

    def get_on_times(self):
        """
        Gets the times at which the pump is supposed to be turned on given by :ref: StartInterval.pump_on_time()

        :return: A list of pump starts in seconds
        :rtype List[int]
        """
        return [p_op for p_op in self.controls if p_op.operation == 1]

    def get_off_times(self):
        """
        Gets the times at which the pump is supposed to be shutdown given by :ref: StartInterval.pump_on_time()

        :return: A list of pump shutdowns in seconds
        :rtype List[int]
        """
        return [p_op for p_op in self.controls if p_op.operation == 0]

    def get_corresponding_tank(self):
        """
        Which pump has a tank associated with it. This method returns the epanet id of that tank.

        :return: The id of the tank in a byte string
        :rtype bytes
        """
        pump_assign_dict = constants.CORRESPONDING_TANK_DICT

        assert self.en_id in pump_assign_dict, f"Pump {self.en_id} does not have corresponding tank!"

        return pump_assign_dict[self.en_id]

    def append_start_power(self, power: float, epanet_timestamp: float):
        """
        Append a power reading and its timestamp to a corresponding start interval. The reading is assign to an
        interval if interval_start <= timestamp < interval_end

        :param power: The power in kW/h
        :type power: float
        :param epanet_timestamp: The timestamp associated to the power reading in seconds
        :type epanet_timestamp: float
        """

        self.powers_reads.append((power, epanet_timestamp))

    def append_start_flow(self, flow: float, epanet_timestamp: float):
        """
        Append a flow reading and its timestamp to a corresponding start interval. The reading is assign to an
        interval if interval_start <= timestamp < interval_end

        :param flow: The flow in m^3/h
        :type flow: float
        :param epanet_timestamp: The timestamp associated to the flow reading in seconds
        :type epanet_timestamp: float
        """

        self.flow_reads.append((flow, epanet_timestamp))

    def calculate_total_pump_volume(self) -> float:
        """
        Computes the total pump flow that was pumped by the pump

        :return: The flow in m^3/h
        :rtype float
        """
        volume_sum = 0
        for interval in self.flow_reads:
            # volume_sum += interval.calculate_volume() TODO: finish this
            pass

        assert volume_sum >= 0

        return volume_sum

    def calculate_energy(self) -> tuple:
        cost_sum = 0
        energy_sum = 0
        for t in self.powers_reads:  # TODO: finish this
            pass

        #     energy_sum += interval_energy
        #
        # self.energy = energy_sum

        assert self.energy >= 0

        return self.energy


class SimulationValve(EpanetElement):
    def __init__(self, en_id: Union[str, bytes], en_index: int):
        super().__init__(en_id, en_index)


class Tank(EpanetElement):
    """
    Class representing a Tank in the simulation
    """
    id: int
    en_index: int
    last_level: float
    simulation_levels: List[float]
    simulation_times: List

    def __init__(self, t_id, en_index, last_level):
        super().__init__(t_id, en_index)
        self.last_level = last_level
        self.simulation_levels = None
        self.simulation_times = None

    def __str__(self):
        return f'TankID: {self.id} | EN_index: {self.en_index} |' \
               'Last:{self.last_level}'


class SimulationJunction(EpanetElement):
    """
    Class representing a Junction in the simulation
    """
    id: int
    en_index: int
    pattern_demand: List[float]
    pattern_index: int

    def __init__(self, j_id, en_index, pattern_demand, pattern_index):
        super().__init__(j_id, en_index)
        self.pattern_demand = pattern_demand
        self.pattern_index = pattern_index


class SimulationPipe(EpanetElement):
    def __init__(self, en_id: Union[str, bytes], en_index: int):
        super().__init__(en_id, en_index)


class Simulation:

    def __init__(self, epanet_file, tanks_info, demands, simulation_duration, n_controls):
        self.tanks = {}
        self.constraints = {}
        self.pumps = {}
        self.valves = {}
        self.pipes = {}
        self.junctions = {}

        self.sim_window_seconds = simulation_duration
        self.N_CONTROLS = n_controls
        self.file = self.render_template(epanet_file, self.N_CONTROLS)
        epamodule.ENopen(self.file, "/dev/null")
        epamodule.ENsettimeparam(epamodule.EN_DURATION, self.sim_window_seconds)
        epamodule.ENsetoption(epamodule.EN_TRIALS, constants.EPANET_MAX_TRIALS)
        # self.save_inp_file()
        self.set_tanks(tanks_info)
        self.set_junctions(demands)
        self._set_links()

        self.cost = 0
        self.energy = 0

    def __str__(self):
        str_constraints = ""
        for t in self.tanks:
            if t in self.constraints:
                u_const = self.constraints[t]['upper_const']
                l_const = self.constraints[t]['lower_const']
                str_constraints += f"\t\tTank: \n\t\t\tupper constr: {u_const}\n\t\t\tlower constr: {l_const}"

        return f'Simulation: \n' \
               f'\tPumps: {len(self.pumps)}\n ' \
               f'\tTanks: {len(self.tanks)}' \
               f'\tJunctions: {len(self.junctions)}' \
               f'\tConstraints:\n' + str_constraints

    @staticmethod
    def render_template(template_name: str, n_controls: int) -> str:
        """
        Convert the {}_server.inp template file that is loaded to the server to a {}.inp file that is interpretable by EPANET
        :param template_name: {}_server.inp
        :param n_controls: number of controls of the type "LINK link_id value AT TIME time HOURS" in the final file
        :return: path of the converted .inp file
        """
        controls_var = constants.DEFAULT_CONTROL * n_controls

        with open(template_name, "r") as f:
            template = Template(f.read())
        inp = template.render(simulation=True, controls=controls_var)
        handle, path = tempfile.mkstemp()
        f = os.fdopen(handle, mode='w')
        f.write(inp)
        f.close()
        return path

    @staticmethod
    def save_inp_file(name: str = f'/tmp/{datetime.datetime.now()}'):
        """
        Saves the INP file corresponding to the network represented by self
        :param name: Path where to save the inp file (Default: '/tmp/{datetime.datetime.now()}')
        :return:
        """
        epamodule.ENsaveinpfile(name)  # THIS IS OPTIONAL
        logger.debug(name)

    def _set_links(self):
        self.pumps = {}
        n_links = epamodule.ENgetcount(epamodule.EN_LINKCOUNT)
        for link_index in range(1, n_links + 1):
            # TODO: apply a creational pattern
            type_ = epamodule.ENgetlinktype(link_index)
            id_ = epamodule.ENgetlinkid(link_index)
            if type_ == epamodule.EN_PUMP:
                p = SimulationPump(id_, link_index)
                self.pumps[p.en_id.decode("utf-8")] = p
            elif type_ == epamodule.EN_FCV or type_ == epamodule.EN_TCV:
                v = SimulationValve(id_, link_index)
                self.valves[v.en_id.decode("utf-8")] = v
            elif type_ == epamodule.EN_PIPE:
                p = SimulationPipe(id_, link_index)
                self.pipes[p.en_id.decode("utf-8")] = p

    def set_junctions(self, demands):
        self.junctions = {}
        for junction_ in demands:
            index = epamodule.ENgetnodeindex(str(junction_))
            pattern_index = epamodule.ENgetpatternindex(f'PatternDemand{junction_}')
            j = SimulationJunction(junction_, index, demands[junction_], pattern_index)
            self.junctions[junction_] = j

            epamodule.ENsetpattern(pattern_index, j.pattern_demand)
            epamodule.ENsetnodevalue(index, epamodule.EN_BASEDEMAND, constants.EPANET_DEFAULT_BASEDEMAND)
            epamodule.ENsetnodevalue(index, epamodule.EN_PATTERN, pattern_index)

    def set_tanks(self, tank_info):
        self.tanks = {}
        for _tank in tank_info:
            index = epamodule.ENgetnodeindex(str(_tank))
            t = Tank(_tank, index, tank_info[_tank])
            self.tanks[t.en_id] = t

    def get_constraints(self):
        upper_constraint = []
        lower_constraint = []
        for _id, tank in self.tanks.items():
            if _id in self.constraints:
                upper_constraint.extend(self.constraints[tank.id]['upper_const'])
                lower_constraint.extend(self.constraints[tank.id]['lower_const'])
        return np.array(upper_constraint).ravel(), np.array(lower_constraint).ravel()

    def set_tank_constraints(self, tank: Tank, upper: list, lower: list):
        assert tank is not None and upper and lower
        self.constraints[tank.id] = {'lower_const': lower,
                                     'upper_const': upper}

    def set_tank_initial_levels(self):
        for _id, tank in self.tanks.items():
            tank.simulation_levels = []
            tank.simulation_times = []
            epamodule.ENsetnodevalue(tank.en_index, epamodule.EN_TANKLEVEL, tank.last_level)

    def calc_energy_and_price(self) -> (float, float):
        """
        Calculates the price and the energy of the pumping operations of a given pump_id. The cost of each optimization
        period is given by :ref: StartInterval.get_cost()

        :return: The pumping cost and the energy spent
        :rtype
        """

        cost_sum = 0
        energy_sum = 0
        for pump_id in self.pumps:
            pump_energy, pump_cost = self.pumps[pump_id].calculate_energy_and_cost()
            cost_sum += pump_cost
            energy_sum += pump_energy

            pump_id.append_index = 0

        assert energy_sum >= 0, "The pumping energy cant be negative!"
        assert cost_sum >= 0, "The pumping cost cant be negative!"
        return energy_sum, cost_sum

    @staticmethod
    def _check_clock_time(clock, pump_times, tank_times, max_clock):
        return clock in pump_times, clock in tank_times, clock == max_clock

    def _get_parallel_stop_times(self, pump_id1) -> set:
        parallel_stop_times = set()
        for pump_id2 in self.pumps:
            if pump_id1 != pump_id2 and self.pumps[pump_id2].get_corresponding_tank() == self.pumps[pump_id1].get_corresponding_tank():
                parallel_stop_times.update(self.pumps[pump_id2].get_epanet_off_times())
        return parallel_stop_times

    def _interval_read_times(self, interval=constants.READING_FREQ_TANKS_WITH_NO_CONTROLS):
        return [interval * i for i in range(int(self.sim_window_seconds / interval)+1)]

    def _create_stop_criterion_data_structures(self):
        _read_times = set()
        _pumpless_tanks_reads = set()
        pump_on_off_times = {}
        for pump_id in self.pumps:
            p_on_times = self.pumps[pump_id].get_on_times()
            p_off_times = self.pumps[pump_id].get_off_times()
            _read_times.update(p_on_times)
            _read_times.update(p_off_times)
            parallel_stop_times = self._get_parallel_stop_times(pump_id)

            pump_on_off_times[pump_id] = {'on_times': set(p_on_times), 'off_times': set(p_off_times),
                                          'parallel_pump_stops': parallel_stop_times}
        _pumpless_tanks_reads.update(self._interval_read_times())
        return _read_times, _pumpless_tanks_reads, pump_on_off_times

    def _get_tanks_without_pumps(self) -> list:
        pumpless_tanks = []
        for tank_id in self.tanks:
            for pump_id in self.pumps:
                if tank_id == self.pumps[pump_id].get_corresponding_tank():
                    break
            else:
                pumpless_tanks.append(tank_id)
        return pumpless_tanks

    def __collect_external_data__(self, collector_func, **func_args):
        collector_func(tanks=self.tanks, pumps=self.pumps, **func_args)

    def _set_controls(self, control_operations: dict):
        """
        Sets the controls on the epanet simulation module. This means that the times at which the pump will be
        turned on and off

        """
        control_index = 1
        for id, operations in control_operations.items():
            link = self.pumps[id] if id in self.pumps else self.valves[id] if id in self.valves else self.pipes[id]
            for op in operations:
                epamodule.ENsetcontrol(control_index,
                                       epamodule.EN_TIMER,
                                       link.en_index,
                                       op[0],  # operation setting
                                       0,
                                       op[1])  # operation time
                control = epamodule.ENgetcontrol(control_index)
                epanet_control_time = int(control[4])
                link.add_control_operation(epanet_control_time, op[0])

                control_index += 1

    @staticmethod
    def _read_pump_power_and_flow(clock_time, sim_pump):
        pump_power = epamodule.ENgetlinkvalue(sim_pump.en_index, epamodule.EN_ENERGY)
        pump_flow = epamodule.ENgetlinkvalue(sim_pump.en_index, epamodule.EN_FLOW)

        sim_pump.append_start_power(pump_power, clock_time)
        sim_pump.append_start_flow(pump_flow, clock_time)

    @staticmethod
    def _read_tank_level_and_time(clock_time, tank):
        level = epamodule.ENgetnodevalue(tank.en_index, epamodule.EN_PRESSURE)
        tank.simulation_levels += [level]
        tank.simulation_times += [clock_time]

    def _reads_on_control_times(self, clock_time, pump_read_times, tank_read_times, pump_on_off_times, pumpless_tanks):
        is_pump_read_time, is_tank_read_time, is_clock_max = self._check_clock_time(clock_time, pump_read_times,
                                                                                    tank_read_times,
                                                                                    self.sim_window_seconds)
        if is_pump_read_time or is_clock_max:
            for pump_id in self.pumps:
                tank_id = self.pumps[pump_id].get_corresponding_tank()
                tank = self.tanks.get(tank_id)
                if tank is not None:
                    if clock_time in pump_on_off_times[pump_id]['on_times']:
                        self._read_pump_power_and_flow(clock_time, self.pumps[pump_id])

                        # self.pumps[pump_id].clocktime += [clock_time]

                        self._read_tank_level_and_time(clock_time, tank)

                    if clock_time in pump_on_off_times[pump_id]['off_times'] or is_clock_max:
                        # self.pumps[pump_id].clocktime += [clock_time]
                        self._read_tank_level_and_time(clock_time, tank)

                    if clock_time in pump_on_off_times[pump_id]['parallel_pump_stops']:
                        pump_power = epamodule.ENgetlinkvalue(self.pumps[pump_id].en_index, epamodule.EN_ENERGY)
                        pump_flow = epamodule.ENgetlinkvalue(self.pumps[pump_id].en_index, epamodule.EN_FLOW)

                        if pump_power > 0:
                            self.pumps[pump_id].append_start_power(pump_power, clock_time)
                            self.pumps[pump_id].append_start_flow(pump_flow, clock_time)

                        # self.pumps[pump_id].clocktime += [clock_time]

        if pumpless_tanks and is_tank_read_time:
            for tank_id in pumpless_tanks:
                self._read_tank_level_and_time(clock_time, self.tanks[tank_id])

    def _read_on_intervals(self, clock_time, read_intervals):
        if clock_time in [time for time in read_intervals if math.isclose(clock_time, time, rel_tol=5)]:
            for pump_id in self.pumps:
                self._read_pump_power_and_flow(clock_time, self.pumps[pump_id])

            for tank_id in self.tanks:
                self._read_tank_level_and_time(clock_time, self.tanks[tank_id])

    @lru_cache(maxsize=0)
    def new_simulation(self, control_operations: dict, data_read_frequency: Union[int, None] = None) -> SimulationResults:
        assert control_operations and data_read_frequency > 0 or None
        pump_read_times, tank_read_times, pump_on_off_times, pumpless_tanks, read_intervals = None, None, None, None, None
        epamodule.ENopenH()

        self.set_tank_initial_levels()

        self._set_controls(control_operations)

        epamodule.ENinitH(10)

        cond = True

        if data_read_frequency is None:
            pumpless_tanks = self._get_tanks_without_pumps()
            pump_read_times, tank_read_times, pump_on_off_times = self._create_stop_criterion_data_structures()
        else:
            read_intervals = self._interval_read_times(data_read_frequency)

        while cond:
            clock_time = epamodule.ENrunH()

            if data_read_frequency is None:
                self._reads_on_control_times(clock_time, pump_read_times, tank_read_times, pump_on_off_times,
                                             pumpless_tanks)
            else:
                self._read_on_intervals(clock_time, read_intervals)

            _ = epamodule.ENnextH()
            cond = not (clock_time >= self.sim_window_seconds)

        # self.energy, self.cost = self.calc_energy_and_price()

        epamodule.ENcloseH()
        results = SimulationResults(self.tanks, None, self.pumps)
        return results


def process_control_operations(dataframe):
    dataframe = dataframe.copy()
    control_operations = {}
    start = dataframe.index[0]
    for col in dataframe.columns:
        if col.startswith('P_') or col.startswith('Pipe'):
            if col.startswith('P_'):
                dataframe[col] = dataframe[col].mask(dataframe[col] >= 1, 1)
                dataframe[col] = dataframe[col].mask(dataframe[col] < 1, 0)

            operations_simplified = dataframe[dataframe[col] != dataframe[col].shift(1)][col].to_frame()
            operations_simplified['op_start_seconds'] = \
                operations_simplified.index.map(lambda d: int((d - start).total_seconds()))
            control_operations[col] = operations_simplified.values

    return control_operations


def process_tank_initial_levels(data, index):
    return {res_col: data[res_col][index] for res_col in [_ for _ in data.columns if _.startswith("Res_")]}


def process_demands(data):
    return {dem_col: data[dem_col].values.tolist() for dem_col in [_ for _ in data.columns if _.startswith("PE_")]}


def calculate_total_n_controls(controls: dict):
    n_controls = 0
    for ct_item in controls:
        n_controls += len(controls[ct_item])
    return n_controls


def epanet_simulation(network_file, sim_duration, control_operations, demands, tank_initial_levels, data_read_step=3600):
    control_operations = process_control_operations(control_operations)
    demands_dict = process_demands(demands)

    simulator = Simulation(network_file, tank_initial_levels, demands_dict, sim_duration,
                           calculate_total_n_controls(control_operations))
    res = simulator.new_simulation(control_operations, data_read_step)
    return np.asarray([_ for _ in res.tank_levels.values()], dtype=float).T


def adcl_simulation():
    adcl_processed, _, _, test_size, abs_levels = load_adcl()
    adcl_raw = load_adcl_raw()

    adcl_raw = adcl_raw[adcl_raw.index >= adcl_processed.index[-test_size]]

    tank_levels = process_tank_initial_levels(abs_levels, adcl_processed.index[-test_size-1])

    adcl_processed = adcl_processed[adcl_processed.index >= adcl_processed.index[-test_size]]
    abs_levels = abs_levels[abs_levels.index >= abs_levels.index[-test_size]]

    demands = adcl_processed[["PE_Aveleira", "PE_Albarqueira", "PE_Espinheira"]]

    sim_duration = int((adcl_raw.index[-1] - adcl_raw.index[0]).total_seconds())
    control_operations = process_control_operations(adcl_raw)
    control_operations.update(
        {v: ctls for v, ctls in process_control_operations(adcl_processed).items() if v.startswith("Pipe")})
    demands_dict = process_demands(demands)

    simulator = Simulation("epanet/adcl_no_valve.inp", tank_levels, demands_dict, sim_duration,
                           calculate_total_n_controls(control_operations))
    res = simulator.new_simulation(control_operations, 900)
    true_levels = abs_levels[
        [level_col for level_col in abs_levels.columns if level_col.startswith("Res_")]].values
    epanet_levels = np.asarray([_ for _ in res.tank_levels.values()], dtype=float).T
    n = nash_sutcliffe(tf.convert_to_tensor(true_levels, np.float32), tf.convert_to_tensor(epanet_levels, np.float32))

    # plot_results_lines(true_levels, epanet_levels)
    print(n.numpy())
    return epanet_levels


if __name__ == '__main__':
   adcl_simulation()