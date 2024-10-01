import numpy as np
from scipy import sparse


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
