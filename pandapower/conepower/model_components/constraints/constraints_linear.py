from typing import Tuple

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.constraints.constraints_base import Constraints


class LinearConstraints(Constraints):
    matrix: sparse.csr_matrix
    rhs: np.ndarray

    def __init__(self, matrix: sparse.csr_matrix, rhs: np.ndarray):
        assert matrix.shape[0] == rhs.shape[0]
        nof_constraints = matrix.shape[0]
        Constraints.__init__(self, nof_constraints)
        self.matrix = matrix
        self.rhs = rhs

    def __add__(self, other):
        assert other.matrix.shape[1] == self.matrix.shape[1]
        return LinearConstraints(sparse.vstack((self.matrix, other.matrix), format='csr'),
                                 np.concatenate((self.rhs, other.rhs)))

    def __iadd__(self, other):
        return self + other

    def to_cone_formulation(self) -> Tuple[sparse.coo_matrix, np.ndarray, int]:
        matrix = self.matrix.tocoo()
        vector = self.rhs
        dimension = self.nof_constraints
        return matrix, vector, dimension
