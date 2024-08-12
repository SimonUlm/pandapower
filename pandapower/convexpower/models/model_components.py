from typing import List

import scipy.sparse as sparse
import numpy as np

from pandapower.convexpower.types.variable_type import VariableType


class VariableSet:
    type: VariableType
    size: int
    initial_values: np.ndarray

    def __init__(self, var_type: VariableType,
                 size: int,
                 initial_values_data: np.ndarray,
                 initial_values_allocated_memory: np.ndarray = None):
        # TODO: size should be an argument
        # TODO: rename initial_values into variables or values
        assert size == np.size(initial_values_data)
        self.type = var_type
        self.size = size
        if initial_values_allocated_memory is None:
            self.initial_values = initial_values_data
        else:
            self.initial_values = initial_values_allocated_memory
            np.copyto(self.initial_values, initial_values_data)
    # TODO: Introduce offset to determine the offset within the allocated memory.
    #  In the model classes, rename initial_values into variables.


class BoxConstraintSet:
    type: VariableType
    size: int
    equalities: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    def __init__(self, var_type: VariableType,
                 size: int,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 equalities: np.ndarray = None):
        assert size == np.size(lower_bounds)
        assert size == np.size(upper_bounds)
        self.type = var_type
        self.size = size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        if equalities is not None:
            assert size == np.size(equalities)
            self.equalities = equalities
        else:
            self.equalities = np.empty(size)
            self.equalities[:] = float('nan')


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
