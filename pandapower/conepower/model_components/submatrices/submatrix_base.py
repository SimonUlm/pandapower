import warnings

import numpy as np
from scipy import sparse


class HermitianSubmatrix:
    _complex_size: int  # number of complex-valued variables stored in upper triangle
    _data: sparse.coo_array
    dim: int
    _full_off_diag_to_antisym_upper_tri: sparse.csc_matrix
    _full_off_diag_to_sym_upper_tri: sparse.csc_matrix
    _half_index_matrix: sparse.csr_matrix
    _nof_edges: int
    _offset_complex_to_real: int
    _real_size: int  # number of real-valued variables stored in upper triangle

    @staticmethod
    def _validate_matrix(matrix: sparse.csr_matrix):
        assert type(matrix) is sparse.csr_matrix
        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.shape[0] > 0
        transposed: sparse.csr_matrix = matrix.transpose().tocsr()
        assert np.any(np.equal(matrix.indices, transposed.indices))
        assert np.any(np.equal(matrix.indptr, transposed.indptr))

    @staticmethod
    def _remove_lower_triangle(full_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        half_matrix = full_matrix.copy()
        for row_idx in range(half_matrix.shape[0]):
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
        self._validate_matrix(adjacency_matrix)
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
