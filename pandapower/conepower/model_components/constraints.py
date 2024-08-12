from typing import List

import numpy as np
from scipy import sparse


class LinearEqualityConstraints:
    nof_constraints: int
    matrix: sparse.csr_matrix
    rhs: np.ndarray

    def __init__(self, matrix: sparse.csr_matrix, rhs: np.ndarray):
        assert matrix.shape[0] == rhs.shape[0]
        self.nof_constraints = matrix.shape[0]
        self.matrix = matrix
        self.rhs = rhs

    @classmethod
    def combine_linear_equality_constraints(cls, linear_equality_constraints: List):
        matrices = [constraint.matrix for constraint in linear_equality_constraints]
        vectors = [constraint.rhs for constraint in linear_equality_constraints]
        new_matrix = sparse.vstack(matrices, 'csr')
        new_rhs = np.concatenate(vectors)
        assert new_matrix.shape[0] == new_rhs.size
        return cls(new_matrix, new_rhs)


class LinearInequalityConstraints:
    npf_constraints: int
    matrix: sparse.csr_matrix
    upper_rhs: np.ndarray

    def __init__(self, matrix: sparse.csr_matrix, rhs: np.ndarray):
        assert matrix.shape[0] == rhs.shape[0]
        self.nof_constraints = matrix.shape[0]
        self.matrix = matrix
        self.upper_rhs = rhs

    # TODO: Klasse ist quasi identisch zu equality constraints.


class SocpConstraintsWithoutConstants:
    nof_constraints: int
    matrices: List[sparse.csr_matrix]
    vectors: List[sparse.csr_matrix]

    def __init__(self, matrices: List[sparse.csr_matrix], vectors: List[sparse.csr_matrix]):
        assert len(matrices) == len(vectors)
        self.nof_constraints = len(matrices)
        dim = matrices[0].shape[1]
        for i in range(len(matrices)):
            assert matrices[i].shape[1] == dim
            assert vectors[i].shape[1] == dim
            assert vectors[i].shape[0] == 1
        self.matrices = matrices
        self.vectors = vectors
