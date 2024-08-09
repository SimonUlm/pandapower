from typing import List

import numpy as np
from cvxopt import solvers
from cvxopt import spmatrix as cvxmatrix
from cvxopt import matrix as cvxvector
from scipy import sparse

from pandapower.convexpower.models.model_components import SocpConstraintsWithoutConstants
from pandapower.convexpower.models.model_socp import ModelSocp


def _matrix_to_cvx(matrix: sparse.csr_matrix) -> cvxmatrix:
    coo = matrix.tocoo()
    return cvxmatrix(coo.data.tolist(),
                     coo.row.tolist(),
                     coo.col.tolist(),
                     size=matrix.shape)


def _vector_to_cvx(vector: np.ndarray) -> cvxvector:
    return cvxvector(vector)


def _socp_to_cvx(socp_constraints: SocpConstraintsWithoutConstants) -> (List[cvxmatrix], List[cvxvector]):
    cvx_matrices = []
    cvx_vectors = [_vector_to_cvx(np.zeros(4)) for i in range(socp_constraints.nof_constraints)]
    for i in range(socp_constraints.nof_constraints):
        matrix = socp_constraints.matrices[i]
        vector = -socp_constraints.vectors[i]
        cvx_matrix = _matrix_to_cvx(sparse.vstack((vector,
                                                   matrix), 'csr'))
        cvx_matrices.append(cvx_matrix)
    return cvx_matrices, cvx_vectors
    # TODO: It works, but it doesn't match the documentation. This needs to be validated thoroughly.


def socp_execute(model: ModelSocp) -> np.ndarray:

    # objective function
    c = _vector_to_cvx(model.linear_cost)

    # inequalities
    Gl = _matrix_to_cvx(model.linear_inequality_constraints.matrix)
    hl = _vector_to_cvx(model.linear_inequality_constraints.upper_rhs)

    # socp
    Gq, hq =_socp_to_cvx(model.socp_constraints)

    # equalities
    A = _matrix_to_cvx(model.linear_equality_constraints.matrix)
    b = _vector_to_cvx(model.linear_equality_constraints.rhs)

    # initial values
    # TODO: Initial values für Slack und Socp werden auch genötigt.

    # solve
    sol = solvers.socp(c,
                       Gl=Gl, hl=hl,
                       Gq=Gq, hq=hq,
                       A=A, b=b)

    # solution vector
    assert sol['x'] is not None
    return np.array(sol['x'])
