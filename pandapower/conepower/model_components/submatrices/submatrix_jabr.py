import math
from typing import List, Tuple

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.submatrices.submatrix_base import HermitianSubmatrix


class JabrSubmatrix(HermitianSubmatrix):

    _lines_to_diag_ff: sparse.csc_matrix
    _lines_to_diag_tt: sparse.csc_matrix
    _lines_to_imag_off_diag_ft: sparse.csc_matrix
    _lines_to_imag_off_diag_tf: sparse.csc_matrix
    _lines_to_real_off_diag_ft: sparse.csc_matrix
    _lines_to_real_off_diag_tf: sparse.csc_matrix
    _nof_lines: int

    def _create_transformation_matrices(self, nodes_from: np.ndarray, nodes_to: np.ndarray):
        # initialize
        self._nof_lines = nodes_from.size
        assert nodes_to.size == self._nof_lines
        assert self._nof_unique_edges <= self._nof_lines
        dummy_diag = sparse.csc_matrix((self._nof_lines, self.dim), dtype=int)
        dummy_off_diag = sparse.csc_matrix((self._nof_lines, self._nof_unique_edges), dtype=int)

        # get how the order of edges relates to the order of variables in submatrix
        mask = nodes_from > nodes_to
        temp_nodes_from = np.where(np.invert(mask), nodes_from, nodes_to)
        temp_nodes_to = np.where(mask, nodes_from, nodes_to)
        lines = np.column_stack((temp_nodes_from, temp_nodes_to))
        line_indices = np.arange(self._nof_lines)
        sorted_line_indices = np.lexsort((temp_nodes_to, temp_nodes_from))
        sorted_lines = lines[sorted_line_indices]
        _, counts = np.unique(sorted_lines, axis=0, return_counts=True)
        data_index_per_line = np.empty(self._nof_lines, dtype=int)
        data_index_per_line[sorted_line_indices] = np.repeat(np.arange(self._nof_unique_edges, dtype=int), counts)

        # diagonal ff
        diag_ff = sparse.coo_matrix((np.ones(self._nof_lines, dtype=int),
                                     (line_indices, nodes_from)),
                                    shape=(self._nof_lines, self.dim))
        self._lines_to_diag_ff = sparse.hstack((diag_ff, dummy_off_diag, dummy_off_diag), format='csc')

        # diagonal tt
        diag_tt = sparse.coo_matrix((np.ones(self._nof_lines, dtype=int),
                                     (line_indices, nodes_to)),
                                    shape=(self._nof_lines, self.dim))
        self._lines_to_diag_tt = sparse.hstack((diag_tt, dummy_off_diag, dummy_off_diag), format='csc')

        # real off-diagonal ft and tf
        real_off_diag_ft = sparse.coo_matrix((np.ones(self._nof_lines, dtype=int),
                                              (line_indices, data_index_per_line)),
                                             shape=(self._nof_lines, self._nof_unique_edges))
        real_off_diag_ft = sparse.hstack((dummy_diag, real_off_diag_ft, dummy_off_diag), format='csc')
        self._lines_to_real_off_diag_ft = real_off_diag_ft
        self._lines_to_real_off_diag_tf = real_off_diag_ft

        # imaginary off-diagonal ft and tf
        off_diag_data = np.ones(self._nof_lines)
        off_diag_data[mask] = -1
        imag_off_diag_ft = sparse.coo_matrix((off_diag_data,
                                              (line_indices, data_index_per_line)),
                                             shape=(self._nof_lines, self._nof_unique_edges))
        imag_off_diag_ft = sparse.hstack((dummy_diag, dummy_off_diag, imag_off_diag_ft), format='csc')
        self._lines_to_imag_off_diag_ft = imag_off_diag_ft
        self._lines_to_imag_off_diag_tf = -imag_off_diag_ft

    def __init__(self,
                 adjacency_matrix: sparse.csr_matrix,
                 nodes_from: np.ndarray,
                 nodes_to: np.ndarray,
                 allocated_data: np.ndarray = None):
        HermitianSubmatrix.__init__(self, adjacency_matrix, allocated_data)
        self._create_transformation_matrices(nodes_from, nodes_to)

    @staticmethod
    def _transform_to_diagonal_block_matrices_without_diagonal(matrix: sparse.csr_matrix)\
            -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        matr_without_diag = matrix.copy()
        matr_without_diag.setdiag(0)
        filtered_rows = [sparse.csr_matrix(matr_without_diag[i][matr_without_diag[i] != 0])
                         for i in range(matr_without_diag.shape[0])]
        diag_block = sparse.block_diag(filtered_rows, 'csr')
        return np.real(diag_block), np.imag(diag_block)

    def create_pg_linear_system_matrix(self, adm_matr: sparse.csr_matrix) -> sparse.csr_matrix:
        adm_real_diag = sparse.diags(np.real(adm_matr.diagonal()))
        real_diag_block, imag_diag_block = self._transform_to_diagonal_block_matrices_without_diagonal(adm_matr)
        real_diag_block = real_diag_block @ self._full_off_diag_to_sym_upper_tri
        imag_diag_block = imag_diag_block @ self._full_off_diag_to_antisym_upper_tri
        return sparse.hstack((adm_real_diag, real_diag_block, imag_diag_block), 'csr')

    def create_qg_linear_system_matrix(self, adm_matr: sparse.csr_matrix) -> sparse.csr_matrix:
        adm_imag_diag = sparse.diags(np.imag(adm_matr.diagonal()))
        real_diag_block, imag_diag_block = self._transform_to_diagonal_block_matrices_without_diagonal(adm_matr)
        real_diag_block = real_diag_block @ self._full_off_diag_to_antisym_upper_tri
        imag_diag_block = imag_diag_block @ self._full_off_diag_to_sym_upper_tri
        return sparse.hstack((-adm_imag_diag, -imag_diag_block, real_diag_block), 'csr')

    def create_jabr_constraints(self) -> Tuple[List[sparse.lil_matrix], List[sparse.lil_matrix]]:
        # initialize lists and calculate offsets
        size = self.dim + self._nof_unique_edges * 2
        matrix_list = [sparse.lil_matrix((3, size), dtype=float) for _ in range(self._nof_unique_edges)]
        vector_list = [sparse.lil_matrix((size, 1), dtype=float) for _ in range(self._nof_unique_edges)]

        # define the matrices and vectors one by one (there has to be a better way...)
        for i in range(self.dim, self._complex_size):
            # preliminary
            row = self._data.row[i]
            col = self._data.col[i]
            # matrix
            matrix = matrix_list[i-self.dim]
            matrix[0, row] = 0.5
            matrix[0, col] = -0.5
            matrix[1, i] = 1
            matrix[2, i+self._offset_complex_to_real] = 1
            # vector
            vector = vector_list[i-self.dim]
            vector[row, 0] = 0.5
            vector[col, 0] = 0.5

        # return matrices and vectors
        return matrix_list, vector_list

    def create_line_apparent_power_constraints(self,
                                               max_apparent_powers: np.ndarray,
                                               y_ff: np.ndarray,
                                               y_ft: np.ndarray,
                                               y_tf: np.ndarray,
                                               y_tt: np.ndarray) ->\
            Tuple[List[sparse.lil_matrix], List[float]]:
        # create two diagonal matrices (one real and one imaginary) for each admittance vector
        real_y_ff = sparse.diags(np.real(y_ff))
        real_y_ft = sparse.diags(np.real(y_ft))
        real_y_tf = sparse.diags(np.real(y_tf))
        real_y_tt = sparse.diags(np.real(y_tt))
        imag_y_ff = sparse.diags(np.imag(y_ff))
        imag_y_ft = sparse.diags(np.imag(y_ft))
        imag_y_tf = sparse.diags(np.imag(y_tf))
        imag_y_tt = sparse.diags(np.imag(y_tt))

        # first and second row of ft constraint for each line, hence we obtain two matrices
        socp_ft_first_rows: sparse.csr_matrix = (real_y_ff @ self._lines_to_diag_ff +
                                                 real_y_ft @ self._lines_to_real_off_diag_ft +
                                                 imag_y_ft @ self._lines_to_imag_off_diag_ft)
        socp_ft_second_rows: sparse.csr_matrix = (-imag_y_ff @ self._lines_to_diag_ff +
                                                  -imag_y_ft @ self._lines_to_real_off_diag_ft +
                                                  real_y_ft @ self._lines_to_imag_off_diag_ft)

        # first and second row of tf constraint for each line, hence we obtain two matrices
        socp_tf_first_rows: sparse.csr_matrix = (real_y_tt @ self._lines_to_diag_tt +
                                                 real_y_tf @ self._lines_to_real_off_diag_tf +
                                                 imag_y_tf @ self._lines_to_imag_off_diag_tf)
        socp_tf_second_rows: sparse.csr_matrix = (-imag_y_tt @ self._lines_to_diag_tt +
                                                  -imag_y_tf @ self._lines_to_real_off_diag_tf +
                                                  real_y_tf @ self._lines_to_imag_off_diag_tf)

        # initialize constraints
        matrix_list = []
        scalar_list = []

        # compose constraints
        for i in range(self._nof_lines):
            # check whether line is unconstrained
            max_power = max_apparent_powers[i]
            if math.isnan(max_power) or math.isinf(max_power) or max_power == 0:
                continue
            # ft
            matrix_list.append(sparse.vstack((socp_ft_first_rows.getrow(i),
                                              socp_ft_second_rows.getrow(i)), format='lil'))
            scalar_list.append(max_power)
            # tf
            matrix_list.append(sparse.vstack((socp_tf_first_rows.getrow(i),
                                              socp_tf_second_rows.getrow(i)), format='lil'))
            scalar_list.append(max_power)

        # noinspection PyTypeChecker
        return matrix_list, scalar_list
