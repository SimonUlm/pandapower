import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.constraints import (LinearEqualityConstraints,
                                                               LinearInequalityConstraints,
                                                               SocpConstraintsWithoutConstants)
from pandapower.conepower.models.model_jabr import ModelJabr
from pandapower.conepower.types.variable_type import VariableType


class ModelSocp:
    linear_cost: np.ndarray
    linear_equality_constraints: LinearEqualityConstraints
    linear_inequality_constraints: LinearInequalityConstraints
    nof_variables: int
    socp_constraints: SocpConstraintsWithoutConstants
    values: np.ndarray

    def __init__(self, nof_variables: int):
        self.nof_variables = nof_variables

    def _box_to_linear_constraints(self, jabr: ModelJabr):

        # initialize
        nof_box_constraints = (jabr.variable_sets[VariableType.PG].size +
                               jabr.variable_sets[VariableType.QG].size +
                               jabr.variable_sets[VariableType.CJJ].size)

        # determine the variables for which the lower bound does not equal the upper bound
        mask = np.concatenate(((jabr.variable_sets[VariableType.PG].lower_bounds !=
                                jabr.variable_sets[VariableType.PG].upper_bounds),
                               (jabr.variable_sets[VariableType.QG].lower_bounds !=
                                jabr.variable_sets[VariableType.QG].upper_bounds),
                               (jabr.variable_sets[VariableType.CJJ].lower_bounds !=
                                jabr.variable_sets[VariableType.CJJ].upper_bounds)))

        # upper bounds
        ub_matrix = sparse.csr_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        ub_matrix.setdiag(1)
        ub_vector = np.concatenate((jabr.variable_sets[VariableType.PG].upper_bounds,
                                    jabr.variable_sets[VariableType.QG].upper_bounds,
                                    jabr.variable_sets[VariableType.CJJ].upper_bounds))

        # lower bounds
        lb_matrix = sparse.csr_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        lb_matrix.setdiag(-1)
        lb_vector = np.concatenate((jabr.variable_sets[VariableType.PG].lower_bounds,
                                    jabr.variable_sets[VariableType.QG].lower_bounds,
                                    jabr.variable_sets[VariableType.CJJ].lower_bounds))

        # combine
        matrix = sparse.vstack((ub_matrix[mask, :],
                                lb_matrix[mask, :]), 'csr')
        vector = np.concatenate((ub_vector[mask], -lb_vector[mask]))
        self.linear_inequality_constraints = LinearInequalityConstraints(matrix, vector)

        # equalities
        mask_inv = np.invert(mask)
        if not mask.any():
            return
        new_equality_constraints = LinearEqualityConstraints(ub_matrix[mask_inv, :], ub_vector[mask_inv])
        self.linear_equality_constraints = (LinearEqualityConstraints
                                            .combine_linear_equality_constraints([new_equality_constraints,
                                                                                  self.linear_equality_constraints]))

    @classmethod
    def from_jabr(cls, jabr: ModelJabr):

        # initialize
        socp = cls(jabr.nof_variables)

        # initial values
        socp.values = jabr.values

        # linear cost
        socp.linear_cost = jabr.linear_cost

        # linear equality constraints
        socp.linear_equality_constraints = (LinearEqualityConstraints
                                            .combine_linear_equality_constraints([jabr.power_flow_equalities,
                                                                                  jabr.hermitian_equalities]))

        # linear inequality constraints
        socp._box_to_linear_constraints(jabr)

        # socp constraints
        socp.socp_constraints = jabr.socp_constraints

        return socp
