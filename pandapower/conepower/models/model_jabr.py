from typing import Dict

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.constraints.constraints_linear import LinearConstraints
from pandapower.conepower.model_components.constraints.constraints_socp import SocpConstraints
from pandapower.conepower.model_components.costs.generator_cost import GeneratorCost
from pandapower.conepower.model_components.costs.quadratic_cost import QuadraticCost
from pandapower.conepower.model_components.submatrices.submatrix_jabr import JabrSubmatrix
from pandapower.conepower.model_components.vector_variable import VariableSet
from pandapower.conepower.models.model_opf import ModelOpf
from pandapower.conepower.types.variable_type import VariableType


class ModelJabr:
    cost: QuadraticCost
    enforce_equalities: bool
    jabr_constraints: SocpConstraints
    line_constraints: SocpConstraints
    power_flow_equalities: LinearConstraints
    nof_variables: int
    submatrix: JabrSubmatrix
    values: np.ndarray
    variable_sets: Dict[VariableType, VariableSet]

    def __init__(self):
        self.nof_variables = 0
        self.variable_sets = {}

    def _transfer_variables_with_box_constraints(self, opf: ModelOpf,
                                                 jabr_variable_type: VariableType,
                                                 starting_index: int) -> int:
        # get original variable type
        opf_variable_type = jabr_variable_type
        if jabr_variable_type is VariableType.CJJ:
            opf_variable_type = VariableType.UMAG

        # get data
        data_dict = {
            'var': opf.variable_sets[opf_variable_type].values,
            'lb': opf.variable_sets[opf_variable_type].lower_bounds,
            'ub': opf.variable_sets[opf_variable_type].upper_bounds,
        }
        if jabr_variable_type is VariableType.CJJ:
            for key, data in data_dict.items():
                data_dict[key] = data ** 2

        # transfer variables
        ending_index = starting_index + opf.variable_sets[opf_variable_type].size
        var_set = VariableSet(data_dict['var'],
                              data_dict['lb'],
                              data_dict['ub'],
                              self.values[starting_index:ending_index])
        self.variable_sets[jabr_variable_type] = var_set

        # return
        return ending_index

    def _add_cjk_variables(self, opf: ModelOpf, starting_index: int) -> int:
        # for now, assume all angles are zero
        assert not opf.variable_sets[VariableType.UANG].values.any()
        ending_index = starting_index + opf.nof_edges
        var_set = VariableSet(values_data=np.ones(opf.nof_edges),
                              values_allocated_memory=self.values[starting_index:ending_index])
        self.variable_sets[VariableType.CJK] = var_set
        return ending_index

    def _add_sjk_variables(self, opf: ModelOpf, starting_index: int) -> int:
        # for now, assume all angles are zero
        assert not opf.variable_sets[VariableType.UANG].values.any()
        ending_index = starting_index + opf.nof_edges
        var_set = VariableSet(values_data=np.zeros(opf.nof_edges),
                              values_allocated_memory=self.values[starting_index:ending_index])
        self.variable_sets[VariableType.SJK] = var_set
        return ending_index

    def _add_active_generator_cost(self, opf: ModelOpf):
        nof_active_power_injections = opf.variable_sets[VariableType.PG].size
        assert opf.active_generator_cost.nof_generators == nof_active_power_injections
        linear_vector = np.zeros(self.nof_variables, dtype=float)
        linear_vector[:nof_active_power_injections] = opf.active_generator_cost.linear_coefficients
        quadratic_vector = np.zeros(self.nof_variables, dtype=float)
        quadratic_vector[:nof_active_power_injections] = opf.active_generator_cost.quadratic_coefficients
        self.cost = QuadraticCost.from_vectors(linear_vector=linear_vector,
                                               quadratic_vector=quadratic_vector)

    def _add_power_flow_equalitites(self, opf: ModelOpf):
        # create matrices for cjj, cjk and sjk
        pg_matrix = self.submatrix.create_pg_linear_system_matrix(opf.admittances.y_bus)
        qg_matrix = self.submatrix.create_qg_linear_system_matrix(opf.admittances.y_bus)

        # create matrices for pg and sg
        connection_matrix = opf.generator_connection_matrix
        pg = sparse.hstack((connection_matrix, sparse.csr_matrix(connection_matrix.shape,
                                                                 dtype=connection_matrix.dtype)))
        qg = sparse.hstack((sparse.csr_matrix(connection_matrix.shape,
                                              dtype=connection_matrix.dtype), connection_matrix))

        # stack matrices
        pf_matrix = sparse.vstack((sparse.hstack((-pg, pg_matrix)),
                                   sparse.hstack((-qg, qg_matrix))),
                                  'csr')

        # add right hand side, consisting of pd and qd
        rhs = -np.concatenate((opf.loads_active, opf.loads_reactive))

        # add to model
        self.power_flow_equalities = LinearConstraints(pf_matrix, rhs)

    def _add_jabr_constraints(self):
        # get constraints without considering the generator variables
        matrix_list, vector_list = self.submatrix.create_jabr_constraints()
        assert len(matrix_list) == len(vector_list)
        # consider generator variables
        nof_gen_variables = self.variable_sets[VariableType.PG].size + self.variable_sets[VariableType.QG].size
        empty_gen_matr = sparse.lil_matrix((3, nof_gen_variables), dtype=float)
        empty_gen_vec = sparse.lil_matrix((nof_gen_variables, 1), dtype=float)
        for i in range(len(matrix_list)):
            matrix_list[i] = sparse.hstack((empty_gen_matr, matrix_list[i]), format='lil', dtype=float)
            vector_list[i] = sparse.vstack((empty_gen_vec, vector_list[i]), format='lil', dtype=float)
        # store in jabr object
        self.jabr_constraints = SocpConstraints(matrix_list, vector_list)

    def _add_line_constraints(self, opf: ModelOpf):
        # get constraints without considering the generator variables
        matrix_list, scalar_list = self.submatrix.create_line_constraints(opf.lines.max_apparent_powers,
                                                                          opf.admittances.y_ff,
                                                                          opf.admittances.y_ft,
                                                                          opf.admittances.y_tf,
                                                                          opf.admittances.y_tt)
        assert len(matrix_list) == len(scalar_list)
        # consider generator variables
        nof_gen_variables = self.variable_sets[VariableType.PG].size + self.variable_sets[VariableType.QG].size
        empty_gen_matr = sparse.lil_matrix((2, nof_gen_variables), dtype=float)
        for i in range(len(matrix_list)):
            matrix_list[i] = sparse.hstack((empty_gen_matr, matrix_list[i]), format='lil', dtype=float)
        # store in jabr object
        self.line_constraints = SocpConstraints(lhs_matrices=matrix_list,
                                                rhs_scalars=scalar_list)

    def _recover_angle_at_bus(self,
                              angles: VariableSet,
                              connectivity_matrix: sparse.csc_matrix,
                              index: int,
                              index_from: int = -1):
        # update
        if index_from == -1:
            angles.values[index] = 0
        else:
            angle = angles.values[index_from] - np.angle(self.submatrix[index_from, index], deg=True)
            angles.values[index] = angle

        # block the way back
        col_start = connectivity_matrix.indptr[index]
        col_end = connectivity_matrix.indptr[index + 1]
        connectivity_matrix.data[col_start:col_end] = -1

        # repeat for all neighbors
        row = connectivity_matrix.getrow(index)
        mask = row.data != -1
        indices_to: np.ndarray = row.indices[mask]
        for idx in indices_to:
            self._recover_angle_at_bus(angles, connectivity_matrix, idx, index)

    def _recover_angles(self, variable_sets: Dict[VariableType, VariableSet]):
        connectivity_matrix = self.submatrix.get_connectivity_matrix()
        self._recover_angle_at_bus(variable_sets[VariableType.UANG], connectivity_matrix, 0)

    @classmethod
    def from_opf(cls, opf: ModelOpf):

        # initialize
        jabr = cls()
        jabr.nof_variables = (opf.variable_sets[VariableType.PG].size +
                              opf.variable_sets[VariableType.QG].size +
                              opf.nof_nodes +
                              opf.nof_edges * 2)

        # variables and box constraints
        jabr.values = np.empty(jabr.nof_variables)
        index = 0
        index = jabr._transfer_variables_with_box_constraints(opf, VariableType.PG, index)
        index = jabr._transfer_variables_with_box_constraints(opf, VariableType.QG, index)
        index = jabr._transfer_variables_with_box_constraints(opf, VariableType.CJJ, index)
        index = jabr._add_cjk_variables(opf, index)
        index = jabr._add_sjk_variables(opf, index)

        # create submatrix
        jabr.submatrix = JabrSubmatrix(opf.admittances.y_bus,
                                       opf.lines.buses_from,
                                       opf.lines.buses_to,
                                       jabr.values[(opf.variable_sets[VariableType.PG].size +
                                                    opf.variable_sets[VariableType.QG].size):
                                                   jabr.nof_variables])

        # save for later whether equality should be enforced for variables with equal lower and upper bound
        jabr.enforce_equalities = opf.enforce_equalities

        # cost
        jabr._add_active_generator_cost(opf)

        # power flow equations
        jabr._add_power_flow_equalitites(opf)

        # jabr constraints
        jabr._add_jabr_constraints()

        # line constraints
        jabr._add_line_constraints(opf)

        return jabr

    def to_opf_variables(self) -> (Dict[VariableType, VariableSet], np.ndarray):

        # initialize
        nof_variables = (self.variable_sets[VariableType.PG].size +
                         self.variable_sets[VariableType.QG].size +
                         self.variable_sets[VariableType.CJJ].size * 2)
        variables = np.empty(nof_variables)
        variable_sets = {}

        # TODO: Fix horrible code below
        # pg
        starting_index = 0
        ending_index = self.variable_sets[VariableType.PG].size
        variable_sets[VariableType.PG] = VariableSet(values_data=self.variable_sets[VariableType.PG].values,
                                                     values_allocated_memory=variables[starting_index:ending_index])

        # qg
        starting_index = ending_index
        ending_index = starting_index + self.variable_sets[VariableType.QG].size
        variable_sets[VariableType.QG] = VariableSet(values_data=self.variable_sets[VariableType.QG].values,
                                                     values_allocated_memory=variables[starting_index:ending_index])

        # umag
        starting_index = ending_index
        ending_index = starting_index + self.variable_sets[VariableType.CJJ].size
        variable_sets[VariableType.UMAG] = VariableSet(values_data=np.sqrt(self.variable_sets[VariableType.CJJ].values),
                                                       values_allocated_memory=variables[starting_index:ending_index])

        # uang TODO: Only works for bus 0 as the ref bus with angle 0!!!
        starting_index = ending_index
        ending_index = starting_index + self.variable_sets[VariableType.CJJ].size
        variable_sets[VariableType.UANG] = VariableSet(values_data=np.empty(self.variable_sets[VariableType.CJJ].size),
                                                       values_allocated_memory=variables[starting_index:ending_index])
        self._recover_angles(variable_sets)

        return variable_sets, variables
