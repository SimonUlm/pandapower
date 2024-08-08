import warnings
from typing import Dict

import scipy.sparse as sparse
import numpy as np

from pandapower.convexpower.model_components import *
from pandapower.convexpower.model_opf import ModelOpf
from pandapower.convexpower.variable_type import VariableType


class ModelJabr:
    box_constraint_sets: Dict[VariableType, BoxConstraintSet]  # erst bei socp raushauen
    cjk_indices_matrix: sparse.coo_matrix
    initial_values: np.ndarray
    hermitian_equalities: LinearEqualityConstraints
    linear_cost: np.ndarray
    power_flow_equalities: LinearEqualityConstraints
    # linear_inequality_constraints: LinearInequalityConstraints, erst bei socp
    nof_variables: int
    socp_constraints: SocpConstraintsWithoutConstants
    variable_sets: Dict[VariableType, VariableSet]

    def __init__(self):
        self.nof_variables = 0
        self.variable_sets = {}
        self.box_constraint_sets = {}

    def _create_cjj_indices_matrx(self, opf: ModelOpf):
        # convert to csc format to ease indexing
        adm_matr = opf.complex_admittance_matrix.tocsc(copy=True)

        # remove diagonal
        adm_matr.setdiag(0)
        adm_matr.eliminate_zeros()

        # cast to integer matrix (it does not matter if values are lost)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx_matrix = adm_matr.astype(dtype=np.int64, casting='unsafe', copy=False)

        # create indexing for cjk and sjk
        starting_index = (opf.variable_sets[VariableType.PG].size +
                          opf.variable_sets[VariableType.QG].size +
                          opf.nof_nodes)
        ending_index = starting_index + opf.nof_edges * 2
        assert ending_index - starting_index == idx_matrix.data.size
        idx_matrix.data = np.arange(starting_index, ending_index)

        # convert to coo, since we are particularly interested in the coordinates of the indices
        idx_matrix = idx_matrix.tocoo(copy=True)

        # create indexing for cjj
        starting_index = (opf.variable_sets[VariableType.PG].size +
                          opf.variable_sets[VariableType.QG].size)
        ending_index = starting_index + opf.nof_nodes
        idx_matrix.setdiag(np.arange(starting_index, ending_index))
        self.cjk_indices_matrix = idx_matrix
        # TODO: Die Struktur ist nicht gerade sehr effizient.
        #  Für die Richtung index -> row/column sollten die Indizies sortiert sein und später ausgenutzt werden.

    def _add_pq_variables(self, opf: ModelOpf, starting_index: int) -> int:
        ending_index = starting_index + opf.variable_sets[VariableType.PG].size
        var_set = VariableSet(VariableType.PG,
                              opf.variable_sets[VariableType.PG].size,
                              opf.variable_sets[VariableType.PG].initial_values,
                              self.initial_values[starting_index:ending_index])
        self.variable_sets[VariableType.PG] = var_set
        return ending_index

    def _add_qg_variables(self, opf: ModelOpf, starting_index: int) -> int:
        ending_index = starting_index + opf.variable_sets[VariableType.QG].size
        var_set = VariableSet(VariableType.QG,
                              opf.variable_sets[VariableType.QG].size,
                              opf.variable_sets[VariableType.QG].initial_values,
                              self.initial_values[starting_index:ending_index])
        self.variable_sets[VariableType.QG] = var_set
        return ending_index

    def _add_cjj_variables(self, opf: ModelOpf, starting_index: int) -> int:
        ending_index = starting_index + opf.nof_nodes
        var_set = VariableSet(VariableType.CJJ,
                              opf.nof_nodes,
                              opf.variable_sets[VariableType.UMAG].initial_values ** 2,
                              self.initial_values[starting_index:ending_index])
        self.variable_sets[VariableType.CJJ] = var_set
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

    def _add_pg_box_constraints(self,
                                opf: ModelOpf,
                                starting_index: int,
                                lb_memory: np.ndarray,
                                ub_memory: np.ndarray) -> int:
        ending_index = starting_index + opf.variable_sets[VariableType.PG].size
        box_set = BoxConstraintSet(VariableType.PG,
                                   opf.variable_sets[VariableType.PG].size,
                                   opf.box_constraint_sets[VariableType.PG].lower_bounds,
                                   opf.box_constraint_sets[VariableType.PG].upper_bounds,
                                   lb_memory[starting_index:ending_index],
                                   ub_memory[starting_index:ending_index])
        self.box_constraint_sets[VariableType.PG] = box_set
        return ending_index

    def _add_qg_box_constraints(self,
                                opf: ModelOpf,
                                starting_index: int,
                                lb_memory: np.ndarray,
                                ub_memory: np.ndarray) -> int:
        ending_index = starting_index + opf.variable_sets[VariableType.QG].size
        box_set = BoxConstraintSet(VariableType.QG,
                                   opf.variable_sets[VariableType.QG].size,
                                   opf.box_constraint_sets[VariableType.QG].lower_bounds,
                                   opf.box_constraint_sets[VariableType.QG].upper_bounds,
                                   lb_memory[starting_index:ending_index],
                                   ub_memory[starting_index:ending_index])
        self.box_constraint_sets[VariableType.QG] = box_set
        return ending_index

    def _add_cjj_box_constraints(self,
                                 opf: ModelOpf,
                                 starting_index: int,
                                 lb_memory: np.ndarray,
                                 ub_memory: np.ndarray) -> int:
        ending_index = starting_index + opf.nof_nodes
        box_set = BoxConstraintSet(VariableType.CJJ,
                                   opf.nof_nodes,
                                   opf.box_constraint_sets[VariableType.UMAG].lower_bounds ** 2,
                                   opf.box_constraint_sets[VariableType.UMAG].upper_bounds ** 2,
                                   lb_memory[starting_index:ending_index],
                                   ub_memory[starting_index:ending_index])
        self.box_constraint_sets[VariableType.CJJ] = box_set
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

    @classmethod
    def from_model_opf(cls, opf: ModelOpf):

        # initialize
        jabr = cls()
        jabr.nof_variables = (opf.variable_sets[VariableType.PG].size +
                              opf.variable_sets[VariableType.QG].size +
                              opf.nof_nodes +
                              opf.nof_edges * 4)

        # variables
        jabr.initial_values = np.empty(jabr.nof_variables)
        index = 0
        index = jabr._add_pq_variables(opf, index)
        index = jabr._add_qg_variables(opf, index)
        index = jabr._add_cjj_variables(opf, index)
        index = jabr._add_cjk_variables(opf, index)
        index = jabr._add_sjk_variables(opf, index)

        # box constraints
        nof_box_constraints = (opf.variable_sets[VariableType.PG].size +
                               opf.variable_sets[VariableType.QG].size +
                               opf.nof_nodes)
        lower_bounds = np.empty(nof_box_constraints)
        upper_bounds = np.empty(nof_box_constraints)
        index = 0
        index = jabr._add_pg_box_constraints(opf, index, lower_bounds, upper_bounds)
        index = jabr._add_qg_box_constraints(opf, index, lower_bounds, upper_bounds)
        index = jabr._add_cjj_box_constraints(opf, index, lower_bounds, upper_bounds)

        # cost
        jabr.linear_cost = np.zeros(jabr.nof_variables)
        jabr._add_active_generator_cost(opf)

        # power flow equations
        jabr._add_power_flow_equalitites(opf)

        # hermitian equations
        jabr._create_cjj_indices_matrx(opf)
        jabr._add_hermitian_equalities(opf)

        # socp constraints
        jabr._add_socp_constraints(opf)

        return jabr
