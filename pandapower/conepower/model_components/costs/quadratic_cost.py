from copy import deepcopy

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.constraints.constraints_socp import SocpConstraints


class QuadraticCost:
    quadratic_matrix: sparse.csr_matrix  # TODO: Document that f(x) = x^T * P * x + q^T * x!!!
    linear_vector: np.ndarray

    def __init__(self,
                 linear_vector: np.ndarray,
                 quadratic_matrix: sparse.csr_matrix = None):
        nof_variables = linear_vector.size
        self.linear_vector = linear_vector
        if quadratic_matrix is not None:
            assert quadratic_matrix.shape[0] == nof_variables
            assert quadratic_matrix.shape[0] == nof_variables
            assert np.all((quadratic_matrix - quadratic_matrix.transpose()).data == 0)
            self.quadratic_matrix = quadratic_matrix
        else:
            self.quadratic_matrix = sparse.csr_matrix((nof_variables, nof_variables), dtype=float)

    def _is_quadratic_matrix_diagonal(self) -> bool:
        _, first_occurrences = np.unique(self.quadratic_matrix.indptr[::-1], return_index=True)
        expected_indices = (len(self.quadratic_matrix.indptr) - 1 - first_occurrences)[:-1]
        return np.all(np.equal(expected_indices, self.quadratic_matrix.indices))

    @classmethod
    def from_vectors(cls,
                     linear_vector: np.ndarray,
                     quadratic_vector: np.ndarray = None):
        # linear case
        if quadratic_vector is None:
            return cls(linear_vector)

        # quadratic case
        nof_variables = linear_vector.size
        assert quadratic_vector.size == nof_variables
        diagonal_matrix = sparse.diags(quadratic_vector).tocsr()
        diagonal_matrix.eliminate_zeros()
        return cls(linear_vector,
                   diagonal_matrix)

    def is_linear(self):
        return self.quadratic_matrix.size == 0

    # TODO: typing of return value
    def scale(self, scaling_factor: float):
        new_cost = deepcopy(self)
        new_cost.quadratic_matrix *= scaling_factor
        new_cost.linear_vector *= scaling_factor
        return new_cost

    def to_socp_constraints(self) -> SocpConstraints:
        assert not self.is_linear()
        assert self._is_quadratic_matrix_diagonal()
        nof_variables = self.linear_vector.size
        # lhs matrix
        chol = self.quadratic_matrix.sqrt()
        lhs_matrix_first_row = sparse.hstack((sparse.lil_matrix(-1, shape=(1, 1), dtype=float),
                                              sparse.lil_matrix(self.linear_vector)), format='lil') / 2
        lhs_matrix_other_rows = sparse.hstack((sparse.lil_matrix((nof_variables, 1), dtype=float),
                                               chol), format='lil')
        lhs_matrix = sparse.vstack((lhs_matrix_first_row, lhs_matrix_other_rows), format='lil')
        # lhs_vector
        lhs_vector = sparse.lil_matrix((nof_variables + 1, 1), dtype=float)
        lhs_vector[0, 0] = 0.5
        # rhs vector
        rhs_vector = -lhs_matrix_first_row.transpose(copy=True)
        # rhs_scalar
        rhs_scalar = 0.5
        # compose
        return SocpConstraints(lhs_matrices=[lhs_matrix],
                               lhs_vectors=[lhs_vector],
                               rhs_vectors=[rhs_vector],
                               rhs_scalars=[rhs_scalar])
