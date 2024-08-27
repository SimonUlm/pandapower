import math
import warnings
from typing import List, Tuple

import numpy as np
from scipy import sparse


class HermitianSubmatrix:
    _data: sparse.coo_array
    dim: int
    _full_off_diag_to_antisym_upper_tri: sparse.csc_matrix
    _full_off_diag_to_sym_upper_tri: sparse.csc_matrix
    _half_index_matrix: sparse.csr_matrix
    _nof_edges: int
    _offset_complex_to_real: int
    _complex_size: int  # number of complex-valued variables stored in upper triangle
    _real_size: int  # number of real-valued variables stored in upper triangle

    @staticmethod
    def _validate_symmetric_matrix(sym_matr: sparse.csr_matrix):
        assert type(sym_matr) is sparse.csr_matrix
        assert sym_matr.shape[0] == sym_matr.shape[1]
        assert sym_matr.shape[0] > 0
        # TODO: Check symmetry

    @staticmethod
    def _remove_lower_triangle(full_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        half_matrix = full_matrix.copy()
        for row_idx in range(half_matrix.shape[1]):  # TODO: Verify if I access right dimension
            row_start = half_matrix.indptr[row_idx]
            row_end = half_matrix.indptr[row_idx + 1]
            col_indices = half_matrix.indices[row_start:row_end]
            mask = col_indices < row_idx
            half_matrix.data[row_start:row_end][mask] = 0
            half_matrix.eliminate_zeros()
        return half_matrix

    def __init__(self,
                 adjacency_matrix: sparse.csr_matrix,
                 allocated_data: np.ndarray = None):
        # validate
        self._validate_symmetric_matrix(adjacency_matrix)
        self.dim = adjacency_matrix.shape[0]

        # prepare index matrices (see below)
        full_index_matrix: sparse.csr_matrix = adjacency_matrix.copy()
        full_index_matrix.setdiag(0)
        full_index_matrix.eliminate_zeros()
        half_index_matrix = self._remove_lower_triangle(full_index_matrix)

        # initialize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            half_index_matrix = half_index_matrix.astype(dtype=int, casting='unsafe')
            full_index_matrix = full_index_matrix.astype(dtype=int, casting='unsafe')
        self._nof_edges = half_index_matrix.size
        self._offset_complex_to_real = self._nof_edges
        self._complex_size = self.dim + self._nof_edges
        self._real_size = self.dim + self._nof_edges * 2

        # create half index matrix that indicates where (in the coo array) the respective matrix entries are stored
        starting_index = self.dim
        ending_index = self._complex_size
        half_index_matrix.data = np.arange(starting_index, ending_index)
        self._half_index_matrix = half_index_matrix

        # create data as coo matrix
        if allocated_data is not None:
            assert allocated_data.size == self._real_size
            data = allocated_data
        else:
            # noinspection PyTypeChecker
            data = np.zeros(self._real_size, dtype=float)
        complex_off_diag_coords = half_index_matrix.tocoo()
        row = np.concatenate((np.arange(self.dim), complex_off_diag_coords.row, complex_off_diag_coords.row))
        col = np.concatenate((np.arange(self.dim), complex_off_diag_coords.col, complex_off_diag_coords.col))
        self._data = sparse.coo_array((data, (row, col)), shape=(self.dim, self.dim))

        # create full index matrix that indicates where the respective matrix entries would have been stored
        # if the full matrix were saved instead of only the upper triangle
        starting_index = self.dim
        ending_index = self.dim + self._nof_edges * 2
        full_index_matrix.data = np.arange(starting_index, ending_index)

        # map half index matrix to full index matrix while disregarding the diagonal
        mapping_matrix = sparse.lil_matrix((self._nof_edges, self._nof_edges * 2), dtype=int)
        indices = np.arange(self._nof_edges)
        pos_array = self._remove_lower_triangle(full_index_matrix).data
        pos_array[:] -= self.dim  # disregard diagonal
        neg_array = self._remove_lower_triangle(full_index_matrix.transpose().tocsr()).data
        neg_array[:] -= self.dim  # disregard diagonal
        mapping_matrix[indices, pos_array] = 1
        mapping_matrix[indices, neg_array] = -1
        self._full_off_diag_to_antisym_upper_tri = mapping_matrix.tocsr(copy=True).transpose()
        self._full_off_diag_to_sym_upper_tri = mapping_matrix.tocsr(copy=True).transpose()
        self._full_off_diag_to_sym_upper_tri.data[:] = 1

    def _get_idx(self, i, j) -> (int, bool):
        # validate
        assert i < self.dim
        assert j < self.dim

        # diagonal
        if i == j:
            return i, False

        # upper triangle
        if i < j:
            idx = self._half_index_matrix[i, j]
            assert idx != 0
            return idx, False

        # lower triangle
        if i > j:
            idx = self._half_index_matrix[j, i]
            assert idx != 0
            return idx, True

        assert False

    def __getitem__(self, key: (int, int)) -> complex:
        i, j = key
        idx, conj = self._get_idx(i, j)
        if idx < self.dim:
            return complex(self._data.data[idx], 0)
        elif conj:
            return complex(self._data.data[idx], -self._data.data[idx+self._offset_complex_to_real])
        else:
            return complex(self._data.data[idx], self._data.data[idx+self._offset_complex_to_real])

    def __setitem__(self, key: (int, int), value: complex):
        i, j = key
        idx, conj = self._get_idx(i, j)
        if idx < self.dim:
            assert value.imag == 0
            self._data.data[idx] = value.real
        elif conj:
            self._data.data[idx] = value.real
            self._data.data[idx+self._offset_complex_to_real] = -value.imag
        else:
            self._data.data[idx] = value.real
            self._data.data[idx+self._offset_complex_to_real] = value.imag

    def get_connectivity_matrix(self) -> sparse.csc_matrix:
        matrix = self._half_index_matrix.tocsc() + self._half_index_matrix.transpose()
        matrix.data[:] = 1
        return matrix

    def get_diagonal(self) -> np.ndarray[float]:
        return self._data.data[:self.dim]


class JabrSubmatrix(HermitianSubmatrix):

    _edges_to_diag_ff: sparse.csc_matrix
    _edges_to_diag_tt: sparse.csc_matrix
    _edges_to_imag_off_diag_ft: sparse.csc_matrix
    _edges_to_imag_off_diag_tf: sparse.csc_matrix
    _edges_to_real_off_diag_ft: sparse.csc_matrix
    _edges_to_real_off_diag_tf: sparse.csc_matrix

    def _create_transformation_matrices(self, nodes_from: np.ndarray, nodes_to: np.ndarray):
        # initialize
        assert nodes_from.size == self._nof_edges
        assert nodes_to.size == self._nof_edges
        dummy_diag = sparse.csc_matrix((self._nof_edges, self.dim), dtype=int)
        dummy_off_diag = sparse.csc_matrix((self._nof_edges, self._nof_edges), dtype=int)

        # get how the order of edges relates to the order of variables in submatrix
        mask = nodes_from > nodes_to
        temp_nodes_from = np.where(np.invert(mask), nodes_from, nodes_to)
        temp_nodes_to = np.where(mask, nodes_from, nodes_to)
        sorted_indices = np.lexsort((temp_nodes_to, temp_nodes_from))

        # diagonal ff
        diag_ff = sparse.coo_matrix((np.ones(self._nof_edges, dtype=int),
                                     (np.arange(self._nof_edges), nodes_from)),
                                    shape=(self._nof_edges, self.dim))
        self._edges_to_diag_ff = sparse.hstack((diag_ff, dummy_off_diag, dummy_off_diag), format='csc')

        # diagonal tt
        diag_tt = sparse.coo_matrix((np.ones(self._nof_edges, dtype=int),
                                     (np.arange(self._nof_edges), nodes_to)),
                                    shape=(self._nof_edges, self.dim))
        self._edges_to_diag_tt = sparse.hstack((diag_tt, dummy_off_diag, dummy_off_diag), format='csc')

        # real off-diagonal ft and tf
        real_off_diag_ft = sparse.coo_matrix((np.ones(self._nof_edges, dtype=int),
                                              (np.arange(self._nof_edges), sorted_indices)),
                                             shape=(self._nof_edges, self._nof_edges))
        real_off_diag_ft = sparse.hstack((dummy_diag, real_off_diag_ft, dummy_off_diag), format='csc')
        self._edges_to_real_off_diag_ft = real_off_diag_ft
        self._edges_to_real_off_diag_tf = real_off_diag_ft

        # imaginary off-diagonal ft and tf
        off_diag_data = np.ones(self._nof_edges)
        off_diag_data[mask] = -1
        imag_off_diag_ft = sparse.coo_matrix((off_diag_data,
                                              (np.arange(self._nof_edges), sorted_indices)),
                                             shape=(self._nof_edges, self._nof_edges))
        imag_off_diag_ft = sparse.hstack((dummy_diag, dummy_off_diag, imag_off_diag_ft), format='csc')
        self._edges_to_imag_off_diag_ft = imag_off_diag_ft
        self._edges_to_imag_off_diag_tf = -imag_off_diag_ft

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
        size = self.dim + self._nof_edges * 2
        matrix_list = [sparse.lil_matrix((3, size), dtype=float) for _ in range(self._nof_edges)]
        vector_list = [sparse.lil_matrix((size, 1), dtype=float) for _ in range(self._nof_edges)]

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

    def create_line_constraints(self,
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
        socp_ft_first_rows: sparse.csr_matrix = (real_y_ff @ self._edges_to_diag_ff +
                                                 real_y_ft @ self._edges_to_real_off_diag_ft +
                                                 imag_y_ft @ self._edges_to_imag_off_diag_ft)
        socp_ft_second_rows: sparse.csr_matrix = (-imag_y_ff @ self._edges_to_diag_ff +
                                                  -imag_y_ft @ self._edges_to_real_off_diag_ft +
                                                  real_y_ft @ self._edges_to_imag_off_diag_ft)

        # first and second row of tf constraint for each line, hence we obtain two matrices
        socp_tf_first_rows: sparse.csr_matrix = (real_y_tt @ self._edges_to_diag_tt +
                                                 real_y_tf @ self._edges_to_real_off_diag_tf +
                                                 imag_y_tf @ self._edges_to_imag_off_diag_tf)
        socp_tf_second_rows: sparse.csr_matrix = (-imag_y_tt @ self._edges_to_diag_tt +
                                                  -imag_y_tf @ self._edges_to_real_off_diag_tf +
                                                  real_y_tf @ self._edges_to_imag_off_diag_tf)

        # initialize constraints
        matrix_list = []
        scalar_list = []

        # compose constraints
        for i in range(self._nof_edges):
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
