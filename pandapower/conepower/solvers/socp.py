import numpy as np
from cvxopt import solvers
from cvxopt import spmatrix as cvxmatrix
from cvxopt import matrix as cvxvector
from scipy import sparse

from pandapower.conepower.models.model_socp import ModelSocp


def _sparse_matrix_to_cvxopt(matrix: sparse.coo_matrix) -> cvxmatrix:
    return cvxmatrix(matrix.data.tolist(),
                     matrix.row.tolist(),
                     matrix.col.tolist(),
                     size=matrix.shape)


def _dense_vector_to_cvxopt(vector: np.ndarray) -> cvxvector:
    return cvxvector(vector)


def socp_execute(model: ModelSocp) -> np.ndarray:

    # objective function
    c = _dense_vector_to_cvxopt(model.linear_cost)

    # inequalities
    g_ineq, h_ineq, dim_ineq = model.linear_inequality_constraints.to_cone_formulation()

    # socp
    g_socp, h_socp, dims_socp = model.socp_constraints.to_cone_formulation()

    # combine cone constraints
    g = _sparse_matrix_to_cvxopt(sparse.vstack((g_ineq, g_socp), format='coo'))
    h = _dense_vector_to_cvxopt(np.concatenate((h_ineq, h_socp)))
    dims = {
        'l': dim_ineq,
        'q': dims_socp,
        's': []
    }

    # equalities
    a_eq, b_eq, _ = model.linear_equality_constraints.to_cone_formulation()
    a = _sparse_matrix_to_cvxopt(a_eq)
    b = _dense_vector_to_cvxopt(b_eq)

    # initial values
    # TODO: Initial values für Slack und Socp werden auch genötigt.

    # solve
    sol = solvers.conelp(c=c,
                         G=g, h=h, dims=dims,
                         A=a, b=b)

    # solution vector
    assert sol['x'] is not None
    return np.array(sol['x'])
