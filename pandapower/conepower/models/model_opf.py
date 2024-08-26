import warnings
from typing import Dict

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.admittances import Admittances
from pandapower.conepower.model_components.lines import Lines
from pandapower.conepower.model_components.vector_variable import VariableSet
from pandapower.conepower.types.variable_type import VariableType

from pandapower.pypower.idx_brch import F_BUS, T_BUS, RATE_A
from pandapower.pypower.idx_gen import GEN_STATUS
from pandapower.pypower.makeSbus import _get_Cg, _get_Sload  # TODO: Think of a better way.
from pandapower.pypower.makeYbus import branch_vectors, makeYbus
from pandapower.pypower.opf_model import opf_model


class ModelOpf:
    admittances: Admittances
    enforce_equalities: bool
    generator_connection_matrix: sparse.csr_matrix
    linear_active_generator_cost: np.ndarray
    lines: Lines
    loads_active: np.ndarray
    loads_reactive: np.ndarray
    nof_edges: int
    nof_nodes: int
    nof_variables: int
    values: np.ndarray
    variable_sets: Dict[VariableType, VariableSet]

    def __init__(self):
        self.nof_edges = 0
        self.nof_nodes = 0
        self.nof_variables = 0
        self.variable_sets = {}
        # self.linear_equality_constraints: LinearEqualityConstraints
        # self.linear_inequality_constraints: LinearInequalityConstraints

    @classmethod
    def from_om(cls, om: opf_model, enforce_equalities: bool = False):
        # initialize
        model = cls()
        model.nof_variables = om.var['N']

        # variables
        model.values = np.empty(model.nof_variables)
        for set_name in om.var['order']:
            starting_index = om.var['idx']['i1'][set_name]
            ending_index = om.var['idx']['iN'][set_name]
            var_type = VariableType.from_str(set_name)
            var_set = VariableSet(om.var['data']['v0'][set_name],
                                  om.var['data']['vl'][set_name],
                                  om.var['data']['vu'][set_name],
                                  model.values[starting_index:ending_index])
            model.variable_sets[var_type] = var_set

        # save for later whether equality should be enforced for variables with equal lower and upper bound
        model.enforce_equalities = enforce_equalities

        # admittance matrix and number of nodes and edges
        base_mva, bus, gen, branch = \
            om.ppc["baseMVA"], om.ppc["bus"], om.ppc["gen"], om.ppc["branch"]
        y_tt, y_ff, y_ft, y_tf = branch_vectors(branch, branch.shape[0])
        y_bus, _, _ = makeYbus(base_mva, bus, branch)
        model.admittances = Admittances(y_bus, y_ff, y_ft, y_tf, y_tt)

        model.nof_edges = branch.shape[0]
        model.nof_nodes = bus.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buses_from = branch[:, F_BUS].astype(int)
            buses_to = branch[:, T_BUS].astype(int)
            max_flows = ((branch[:, RATE_A].astype(float) / base_mva) ** 2)
        model.lines = Lines(buses_from, buses_to, max_flows)
        assert model.lines.nof_lines == model.nof_edges

        # loads
        loads = _get_Sload(bus, None) / base_mva
        model.loads_active = np.real(loads)
        model.loads_reactive = np.imag(loads)

        # generator connection matrix
        on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
        gen_on = gen[on, :]
        model.generator_connection_matrix = _get_Cg(gen_on, bus)

        # linear generator cost
        gen_cost = om.ppc['gencost']
        assert np.all(gen_cost[:, 0] == 2)  # polynomial model
        assert np.all(gen_cost[:, 1] == 0)  # no startup cost
        assert np.all(gen_cost[:, 2] == 0)  # no shutdown cost
        assert np.all(gen_cost[:, 3] == 2)  # linear cost function
        assert np.all(gen_cost[:, 5] == 0)  # no affine cost function
        model.linear_active_generator_cost = np.copy(gen_cost[:, 4])
        assert np.size(model.linear_active_generator_cost) == np.size(model.variable_sets[VariableType.PG].size)

        return model
