from typing import Tuple

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.constraints.constraints_base import Constraints


class LinearConstraints(Constraints):
    matrix: sparse.csr_matrix = None
    rhs: np.ndarray = None

    def __init__(self,
                 matrix: sparse.csr_matrix = None,
                 rhs: np.ndarray = None):
        if matrix is None and rhs is None:
            Constraints.__init__(self, 0)
        elif matrix is not None and rhs is not None:
            assert matrix.shape[0] == rhs.shape[0]
            nof_constraints = matrix.shape[0]
            Constraints.__init__(self, nof_constraints)
            self.matrix = matrix
            self.rhs = rhs
        else:
            assert False

    def __add__(self, other):
        if self.is_empty() and other.is_empty():
            return self
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        assert other.matrix.shape[1] == self.matrix.shape[1]
        return LinearConstraints(sparse.vstack((self.matrix, other.matrix), format='csr'),
                                 np.concatenate((self.rhs, other.rhs)))

    def prepend_variable(self):
        self.matrix = sparse.hstack((sparse.csr_matrix((self.matrix.shape[0], 1), dtype=float), self.matrix),
                                    format='csr')

    def to_cone_formulation(self) -> Tuple[sparse.coo_matrix, np.ndarray, int]:
        matrix = self.matrix.tocoo()
        vector = self.rhs
        dimension = self.nof_constraints
        return matrix, vector, dimension
