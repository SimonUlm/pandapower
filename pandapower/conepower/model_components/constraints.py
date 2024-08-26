from typing import List, Tuple

import numpy as np
from scipy import sparse


class LinearConstraints:
    nof_constraints: int
    matrix: sparse.csr_matrix
    rhs: np.ndarray

    def __init__(self, matrix: sparse.csr_matrix, rhs: np.ndarray):
        assert matrix.shape[0] == rhs.shape[0]
        self.nof_constraints = matrix.shape[0]
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


class SocpConstraints:
    nof_constraints: int
    lhs_matrices: List[sparse.lil_matrix]
    lhs_vectors: List[sparse.lil_matrix]
    rhs_scalars: List[float]
    rhs_vectors: List[sparse.lil_matrix]

    def __init__(self,
                 lhs_matrices: List[sparse.lil_matrix],
                 rhs_vectors: List[sparse.lil_matrix] = None,
                 lhs_vectors: List[sparse.lil_matrix] = None,
                 rhs_scalars: List[float] = None):
        # initialize
        self.nof_constraints = len(lhs_matrices)
        var_dim = lhs_matrices[0].shape[1]

        # necessary variables: lhs_matrices
        for i in range(self.nof_constraints):
            assert lhs_matrices[i].shape[1] == var_dim
        self.lhs_matrices = lhs_matrices

        # optional variables: rhs_vectors
        if rhs_vectors is not None:
            assert len(rhs_vectors) == self.nof_constraints
            for i in range(self.nof_constraints):
                assert rhs_vectors[i].shape[0] == var_dim
                assert rhs_vectors[i].shape[1] == 1
            self.rhs_vectors = rhs_vectors
        else:
            self.rhs_vectors = [sparse.lil_matrix((var_dim, 1), dtype=float)
                                for _ in range(self.nof_constraints)]

        # optional variables: lhs_vectors
        if lhs_vectors is not None:
            assert len(lhs_vectors) == self.nof_constraints
            for i in range(self.nof_constraints):
                assert lhs_vectors[i].shape[0] == lhs_matrices[i].shape[0]
                assert lhs_vectors[i].shape[1] == 1
            self.lhs_vectors = lhs_vectors
        else:
            self.lhs_vectors = [sparse.lil_matrix((lhs_matrices[i].shape[0], 1), dtype=float)
                                for i in range(self.nof_constraints)]

        # optional variables: rhs_scalars
        if rhs_scalars is not None:
            assert len(rhs_scalars) == self.nof_constraints
            self.rhs_scalars = rhs_scalars
        else:
            self.rhs_scalars = [0 for _ in range(self.nof_constraints)]

    def __add__(self, other):
        return SocpConstraints(self.lhs_matrices + other.lhs_matrices,
                               self.rhs_vectors + other.rhs_vectors,
                               self.lhs_vectors + other.lhs_vectors,
                               self.rhs_scalars + other.rhs_scalars)

    def __iadd__(self, other):
        return self + other

    def to_cone_formulation(self) -> Tuple[sparse.coo_matrix, np.ndarray, List[int]]:
        matrix = -sparse.vstack([sparse.vstack((self.rhs_vectors[i].transpose(),
                                                self.lhs_matrices[i])) for i in range(self.nof_constraints)],
                                format='coo')
        vectors = sparse.vstack([sparse.vstack((self.rhs_scalars[i],
                                                self.lhs_vectors[i])) for i in range(self.nof_constraints)],
                                format='coo').todense()
        dimensions = [1 + self.lhs_matrices[i].shape[0] for i in range(self.nof_constraints)]
        return matrix, np.squeeze(np.asarray(vectors)), dimensions
