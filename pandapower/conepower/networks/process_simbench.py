import copy
from enum import Enum
from typing import Tuple

import pandas as pd

import pandapower as pp
from pandapower.create import create_empty_network


COST_FACTOR = 1
REF_BUS_GEN_LIMIT = 100
REF_BUS_VM = 1.0


class StudyCase(Enum):
    UNDEFINED = 0
    HIGH_LOAD_HIGH_PV = 1
    LOW_LOAD_HIGH_PV = 2
    HIGH_LOAD_LOW_PV = 3
    LOW_LOAD_LOW_PV = 4

    def to_str(self) -> str:
        if self is self.HIGH_LOAD_HIGH_PV:
            return 'hPV'
        elif self is self.LOW_LOAD_HIGH_PV:
            return 'lPV'
        elif self is self.HIGH_LOAD_LOW_PV:
            return 'h'
        elif self is self.LOW_LOAD_LOW_PV:
            return 'l'
        else:
            assert False


class InverterControlMode(Enum):
    UNDEFINED = 0
    ZERO_Q = 1
    CONST_PHI = 2

    def get_factor(self) -> float:
        if self is self.ZERO_Q:
            return 0.
        elif self is self.CONST_PHI:
            return 0.1
        else:
            assert False


def _get_scaling(loadcases: pd.DataFrame, study_case: StudyCase) -> Tuple[float, float, float]:
    if study_case is StudyCase.HIGH_LOAD_HIGH_PV or study_case is StudyCase.LOW_LOAD_HIGH_PV:
        p_gen_scaling = loadcases.loc[study_case.to_str()]['PV_p']
        p_load_scaling = loadcases.loc[study_case.to_str()]['pload']
        q_load_scaling = loadcases.loc[study_case.to_str()]['qload']
        pass
    elif study_case is StudyCase.HIGH_LOAD_LOW_PV:
        p_gen_scaling = 0.25
        p_load_scaling = 1
        q_load_scaling = 1
    elif study_case is StudyCase.LOW_LOAD_LOW_PV:
        p_gen_scaling = 0.25
        p_load_scaling = 0.1
        q_load_scaling = 0.122543
    else:
        assert False
    return p_gen_scaling, p_load_scaling, q_load_scaling


def _sgen_to_gen(sgen: pd.DataFrame, gen: pd.DataFrame) -> pd.DataFrame:
    common_columns = sgen.columns.intersection(gen.columns)
    gen[common_columns] = sgen[common_columns]
    return gen.assign(vm_pu=1.0, slack=False)


def lv_grid_to_pf(net: pp.pandapowerNet,
                  study_case: StudyCase = StudyCase.HIGH_LOAD_HIGH_PV,
                  mode: InverterControlMode = InverterControlMode.ZERO_Q) -> pp.pandapowerNet:
    # extract data
    bus = net['bus'].copy()
    bus_geodata = net['bus_geodata'].copy()
    ext_grid = net['ext_grid'].copy()
    line = net['line'].copy()
    load = net['load'].copy()
    loadcases = net['loadcases'].copy()
    profiles = copy.deepcopy(net['profiles'])
    sgen = net['sgen'].copy()
    std_types = net['std_types'].copy()
    switch = net['switch'].copy()
    trafo = net['trafo'].copy()

    # extract load case
    # TODO: Profiles verwenden, um weitere Fälle zu erstellen
    p_gen_scaling, p_load_scaling, q_load_scaling = _get_scaling(loadcases, study_case)

    # sgen
    sgen['p_mw'] = sgen['sn_mva'] * p_gen_scaling
    sgen["q_mvar"] = mode.get_factor() * sgen["p_mw"]
    # load
    load['p_mw'] *= p_load_scaling
    load['q_mvar'] *= q_load_scaling
    # ext grid
    ext_grid['vm_pu'] = 1

    # create new net
    pf_net = create_empty_network()
    pf_net['bus'] = bus
    pf_net['bus_geodata'] = bus_geodata
    pf_net['ext_grid'] = ext_grid
    pf_net['line'] = line
    pf_net['load'] = load
    pf_net['sgen'] = sgen
    pf_net['std_types'] = std_types
    pf_net['switch'] = switch
    pf_net['trafo'] = trafo

    return pf_net


def lv_grid_to_opf(net: pp.pandapowerNet,
                   study_case: StudyCase = StudyCase.HIGH_LOAD_HIGH_PV) -> pp.pandapowerNet:
    # extract data
    bus = net['bus'].copy()
    bus_geodata = net['bus_geodata'].copy()
    ext_grid = net['ext_grid'].copy()
    gen = net['gen'].copy()
    line = net['line'].copy()
    load = net['load'].copy()
    loadcases = net['loadcases'].copy()
    profiles = copy.deepcopy(net['profiles'])
    sgen = net['sgen'].copy()
    std_types = net['std_types'].copy()
    switch = net['switch'].copy()
    trafo = net['trafo'].copy()

    # extract load case
    # TODO: Profiles verwenden, um weitere Fälle zu erstellen
    p_gen_scaling, p_load_scaling, q_load_scaling = _get_scaling(loadcases, study_case)

    # modify data
    ext_grid = ext_grid.assign(vm_pu=REF_BUS_VM,
                               max_p_mw=REF_BUS_GEN_LIMIT,
                               min_p_mw=-REF_BUS_GEN_LIMIT,
                               max_q_mvar=REF_BUS_GEN_LIMIT,
                               min_q_mvar=-REF_BUS_GEN_LIMIT)
    assert ext_grid.shape[0] == 1
    # gen
    gen = _sgen_to_gen(sgen, gen)
    gen['max_p_mw'] = gen['p_mw'] = gen['sn_mva'] * p_gen_scaling
    gen['min_p_mw'] = gen['sn_mva'] * 0.2
    gen['max_q_mvar'] = gen['sn_mva'] * 1.1 * 0.44
    gen['min_q_mvar'] = gen['sn_mva'] * 1.1 * (-0.44)
    # load
    load['p_mw'] *= p_load_scaling
    load['q_mvar'] *= q_load_scaling
    # trafo
    # trafo = trafo.assign(pfe_kw=0.0)
    # TODO: Überlege, ob man das wieder rausnehmen sollte, das beeinflusst die charging conductance!
    #       Evtl. sollte man einen verlustfreien Transformer annehmen.
    # ext grid
    ext_grid['vm_pu'] = 1

    # create new net and add cost
    opf_net = create_empty_network()
    opf_net['bus'] = bus
    opf_net['bus_geodata'] = bus_geodata
    opf_net['ext_grid'] = ext_grid
    opf_net['gen'] = gen
    opf_net['line'] = line
    opf_net['load'] = load
    opf_net['std_types'] = std_types
    opf_net['switch'] = switch
    opf_net['trafo'] = trafo
    pp.create_poly_cost(opf_net,
                        element=0,
                        et='ext_grid',
                        cp1_eur_per_mw=COST_FACTOR * 1)
    #for i in range(opf_net['gen'].shape[0]):
    #    pp.create_poly_cost(opf_net,
    #                        element=i,
    #                        et='gen',
    #                        cp1_eur_per_mw=COST_FACTOR * 0.001)

    return opf_net
