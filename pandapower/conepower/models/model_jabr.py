import math
import warnings
from typing import Dict

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.constraints import LinearEqualityConstraints, SocpConstraintsWithoutConstants
from pandapower.conepower.model_components.vector_variables import BoxConstraintSet, VariableSet
from pandapower.conepower.models.model_opf import ModelOpf
from pandapower.conepower.types.variable_type import VariableType


class ModelJabr:
    box_constraint_sets: Dict[VariableType, BoxConstraintSet]
    cjk_indices_matrix: sparse.coo_matrix
    initial_values: np.ndarray
    hermitian_equalities: LinearEqualityConstraints
    linear_cost: np.ndarray
    power_flow_equalities: LinearEqualityConstraints
    nof_variables: int
    socp_constraints: SocpConstraintsWithoutConstants
    variable_sets: Dict[VariableType, VariableSet]

    def __init__(self):
        self.nof_variables = 0
        self.variable_sets = {}
        self.box_constraint_sets = {}

    def _create_cjk_indices_matrx(self, opf: ModelOpf):
        # create indices matrix by analyzing the admittance matrix
        adm_matr = opf.complex_admittance_matrix.copy()

        # remove diagonal
        adm_matr.setdiag(0)
        adm_matr.eliminate_zeros()

        # cast to integer matrix (it does not matter if values are lost)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx_matrix = adm_matr.astype(dtype=np.int64, casting='unsafe', copy=False)

        # write the indices of the variable vector into the matrix
        starting_index = (opf.variable_sets[VariableType.PG].size +
                          opf.variable_sets[VariableType.QG].size +
                          opf.nof_nodes)
        ending_index = starting_index + opf.nof_edges * 2
        assert ending_index - starting_index == idx_matrix.data.size
        idx_matrix.data = np.arange(starting_index, ending_index)

        # convert to coo, since we are particularly interested in the coordinates of the matrix entries
        idx_matrix = idx_matrix.tocoo(copy=True)
        self.cjk_indices_matrix = idx_matrix

    def _transfer_variables_with_box_constraints(self, opf: ModelOpf,
                                                 jabr_variable_type: VariableType,
                                                 starting_index: int) -> int:
        # get original variable type
        opf_variable_type = jabr_variable_type
        if jabr_variable_type is VariableType.CJJ:
            opf_variable_type = VariableType.UMAG

        # get data
        data_dict = {
            'var': opf.variable_sets[opf_variable_type].initial_values,
            'lb': opf.box_constraint_sets[opf_variable_type].lower_bounds,
            'ub': opf.box_constraint_sets[opf_variable_type].upper_bounds,
            'eq': opf.box_constraint_sets[opf_variable_type].equalities
        }
        if jabr_variable_type is VariableType.CJJ:
            for key, data in data_dict.items():
                data_dict[key] = data ** 2

        # transfer variables
        ending_index = starting_index + opf.variable_sets[opf_variable_type].size
        var_set = VariableSet(jabr_variable_type,
                              data_dict['var'].size,
                              data_dict['var'],
                              self.initial_values[starting_index:ending_index])
        self.variable_sets[jabr_variable_type] = var_set

        # transfer box constraints
        box_set = BoxConstraintSet(jabr_variable_type,
                                   data_dict['var'].size,
                                   data_dict['lb'],
                                   data_dict['ub'],
                                   data_dict['eq'])
        self.box_constraint_sets[jabr_variable_type] = box_set

        # return
        return ending_index

    def _add_cjk_variables(self, opf: ModelOpf, starting_index: int) -> int:
        # for now, assume all angles are zero
        assert not opf.variable_sets[VariableType.UANG].initial_values.any()
        ending_index = starting_index + opf.nof_edges * 2
        var_set = VariableSet(VariableType.CJK,
                              opf.nof_edges * 2,
                              np.ones(opf.nof_edges * 2),
                              self.initial_values[starting_index:ending_index])
        self.variable_sets[VariableType.CJK] = var_set
        return ending_index

    def _add_sjk_variables(self, opf: ModelOpf, starting_index: int) -> int:
        # for now, assume all angles are zero
        assert not opf.variable_sets[VariableType.UANG].initial_values.any()
        ending_index = starting_index + opf.nof_edges * 2
        var_set = VariableSet(VariableType.SJK,
                              opf.nof_edges * 2,
                              np.zeros(opf.nof_edges * 2),
                              self.initial_values[starting_index:ending_index])
        self.variable_sets[VariableType.SJK] = var_set
        return ending_index

    def _add_active_generator_cost(self, opf: ModelOpf):
        nof_active_power_injections = opf.variable_sets[VariableType.PG].size
        assert opf.linear_active_generator_cost.size == nof_active_power_injections
        self.linear_cost[0:nof_active_power_injections] += opf.linear_active_generator_cost

    def _add_power_flow_equalitites(self, opf: ModelOpf):

        # initialize
        adm_matr = opf.complex_admittance_matrix

        # get diagonals for cjj
        real_diag = sparse.diags(np.real(adm_matr.diagonal()))
        imag_diag = sparse.diags(np.imag(adm_matr.diagonal()))

        # create block matrices for cjk and sjk
        adm_matr_without_diag = adm_matr.copy()
        adm_matr_without_diag.setdiag(0)
        filtered_rows = [sparse.csc_matrix(adm_matr_without_diag[i][adm_matr_without_diag[i] != 0])
                         for i in range(adm_matr_without_diag.shape[0])]
        block_diag = sparse.block_diag(filtered_rows)
        real_block_diag = np.real(block_diag)
        imag_block_diag = np.imag(block_diag)

        # create matrices for injection variables
        connection_matrix = opf.generator_connection_matrix
        pg = sparse.hstack((connection_matrix, sparse.csr_matrix(connection_matrix.shape,
                                                                 dtype=connection_matrix.dtype)))
        qg = sparse.hstack((sparse.csr_matrix(connection_matrix.shape,
                                              dtype=connection_matrix.dtype), connection_matrix))

        # stack matrices
        pf_matrix = sparse.vstack((sparse.hstack((-pg, real_diag, real_block_diag, imag_block_diag)),     # pg
                                   sparse.hstack((-qg, -imag_diag, -imag_block_diag, real_block_diag))),  # qg
                                  'csr')
        rhs = -np.concatenate((opf.loads_active, opf.loads_reactive))

        # add to model
        self.power_flow_equalities = LinearEqualityConstraints(pf_matrix, rhs)

    def _add_hermitian_equalities(self, opf: ModelOpf):

        # get relevant matrices
        coo_matrix = self.cjk_indices_matrix
        csr_matrix = coo_matrix.tocsr()

        # get cjk (or sjk) in the upper triangle
        mask = coo_matrix.row < coo_matrix.col
        count = mask.sum()
        rows = coo_matrix.row[mask]
        columns = coo_matrix.col[mask]

        # initialize matrices and calculate offset between cjk and sjk indices
        cjk_matrix = sparse.csr_matrix((count, self.nof_variables), dtype=np.float64)
        sjk_matrix = sparse.csr_matrix((count, self.nof_variables), dtype=np.float64)
        offset = opf.nof_edges * 2

        # define the equations one by one (there has to be a better way...)
        for i in range(count):
            row = rows[i]
            col = columns[i]
            cjk_matrix[i, csr_matrix[row, col]] = 1
            cjk_matrix[i, csr_matrix[col, row]] = -1
            sjk_matrix[i, csr_matrix[row, col] + offset] = 1
            sjk_matrix[i, csr_matrix[col, row] + offset] = 1

        # stack together and define rhs
        matrix = sparse.vstack((cjk_matrix,
                                sjk_matrix), 'csr')
        rhs = np.zeros(count * 2)
        self.hermitian_equalities = LinearEqualityConstraints(matrix, rhs)

    def _add_socp_constraints(self, opf:ModelOpf):

        # get relevant matrices
        coo_matrix = self.cjk_indices_matrix
        csr_matrix = coo_matrix.tocsr()

        # get cjk (or sjk) in the upper triangle
        mask = coo_matrix.row < coo_matrix.col
        count = mask.sum()
        rows = coo_matrix.row[mask]
        columns = coo_matrix.col[mask]

        # TODO: Duplicated code.

        # initialize lists and calculate offsets
        matrix_list = [sparse.csr_matrix((3, self.nof_variables), dtype=np.float64) for i in range(count)]
        vector_list = [sparse.csr_matrix((1, self.nof_variables), dtype=np.float64) for i in range(count)]
        offset_to_cjj = opf.variable_sets[VariableType.PG].size + opf.variable_sets[VariableType.QG].size
        offset_cjk_to_sjk = opf.nof_edges * 2

        # define the matrices and vectors one by one (there has to be a better way...)
        for i in range(count):
            # preliminary
            row = rows[i]
            col = columns[i]
            # matrix
            matrix = matrix_list[i]
            matrix[0, row + offset_to_cjj] = 0.5
            matrix[0, col + offset_to_cjj] = -0.5
            matrix[1, csr_matrix[row, col]] = 1
            matrix[2, csr_matrix[row, col] + offset_cjk_to_sjk] = 1
            # vector
            vector = vector_list[i]
            vector[0, row + offset_to_cjj] = 0.5
            vector[0, col + offset_to_cjj] = 0.5

        # assign to object
        self.socp_constraints = SocpConstraintsWithoutConstants(matrix_list, vector_list)

    def _recover_angle_at_bus(self,
                              new_angles: VariableSet,
                              connectivity_matrix: sparse.csc_matrix,
                              index: int,
                              index_from: int = -1):

        # update
        if index_from == -1:
            new_angles.initial_values[index] = 0
        else:
            index_cjk_sjk = connectivity_matrix[index_from, index]
            angle = new_angles.initial_values[index_from] - math.atan2(self.variable_sets[VariableType.SJK]
                                                                       .initial_values[index_cjk_sjk],
                                                                       self.variable_sets[VariableType.CJK]
                                                                       .initial_values[index_cjk_sjk])
            new_angles.initial_values[index] = angle

        # block the way back
        col_start = connectivity_matrix.indptr[index]
        col_end = connectivity_matrix.indptr[index + 1]
        connectivity_matrix.data[col_start:col_end] = -1

        # repeat for all neighbors
        row = connectivity_matrix.getrow(index)
        mask = row.data != -1
        indices_to: np.ndarray = row.indices[mask]
        for idx in indices_to:
            self._recover_angle_at_bus(new_angles, connectivity_matrix, idx, index)
        pass

    def _recover_angles_from_radial_network(self, variable_sets: Dict[VariableType, VariableSet]):

        # initialize
        connectivity_matrix: sparse.csc_matrix = self.cjk_indices_matrix.tocsc(copy=True)
        connectivity_matrix.setdiag(0)  # temp
        connectivity_matrix.eliminate_zeros()  # temp
        offset = (self.variable_sets[VariableType.PG].size +
                  self.variable_sets[VariableType.QG].size +
                  self.variable_sets[VariableType.CJJ].size)
        connectivity_matrix.data -= offset

        # start with bus 0
        self._recover_angle_at_bus(variable_sets[VariableType.UANG], connectivity_matrix, 0)

    @classmethod
    def from_opf(cls, opf: ModelOpf):

        # initialize
        jabr = cls()
        jabr.nof_variables = (opf.variable_sets[VariableType.PG].size +
                              opf.variable_sets[VariableType.QG].size +
                              opf.nof_nodes +
                              opf.nof_edges * 4)

        # variables and box constraints
        jabr.initial_values = np.empty(jabr.nof_variables)
        index = 0
        index = jabr._transfer_variables_with_box_constraints(opf, VariableType.PG, index)
        index = jabr._transfer_variables_with_box_constraints(opf, VariableType.QG, index)
        index = jabr._transfer_variables_with_box_constraints(opf, VariableType.CJJ, index)
        index = jabr._add_cjk_variables(opf, index)
        index = jabr._add_sjk_variables(opf, index)

        # cost
        jabr.linear_cost = np.zeros(jabr.nof_variables)
        jabr._add_active_generator_cost(opf)

        # power flow equations
        jabr._add_power_flow_equalitites(opf)

        # hermitian equations
        jabr._create_cjk_indices_matrx(opf)
        jabr._add_hermitian_equalities(opf)

        # socp constraints
        jabr._add_socp_constraints(opf)

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
        variable_sets[VariableType.PG] = VariableSet(VariableType.PG,
                                                     self.variable_sets[VariableType.PG].size,
                                                     self.variable_sets[VariableType.PG].initial_values,
                                                     variables[starting_index:ending_index])

        # qg
        starting_index = ending_index
        ending_index = starting_index + self.variable_sets[VariableType.QG].size
        variable_sets[VariableType.QG] = VariableSet(VariableType.QG,
                                                     self.variable_sets[VariableType.QG].size,
                                                     self.variable_sets[VariableType.QG].initial_values,
                                                     variables[starting_index:ending_index])

        # umag
        starting_index = ending_index
        ending_index = starting_index + self.variable_sets[VariableType.CJJ].size
        variable_sets[VariableType.UMAG] = VariableSet(VariableType.UMAG,
                                                       self.variable_sets[VariableType.CJJ].size,
                                                       np.sqrt(self.variable_sets[VariableType.CJJ].initial_values),
                                                       variables[starting_index:ending_index])

        # uang TODO: Only works for bus 0 as the ref bus with angle 0!!!
        starting_index = ending_index
        ending_index = starting_index + self.variable_sets[VariableType.CJJ].size
        variable_sets[VariableType.UANG] = VariableSet(VariableType.UANG,
                                                       self.variable_sets[VariableType.CJJ].size,
                                                       np.empty(self.variable_sets[VariableType.CJJ].size),
                                                       variables[starting_index:ending_index])
        self._recover_angles_from_radial_network(variable_sets)

        return variable_sets, variables
