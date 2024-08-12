import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.constraints import (LinearEqualityConstraints,
                                                               LinearInequalityConstraints,
                                                               SocpConstraintsWithoutConstants)
from pandapower.conepower.models.model_jabr import ModelJabr
from pandapower.conepower.types.variable_type import VariableType


class ModelSocp:
    initial_values: np.ndarray
    linear_cost: np.ndarray
    linear_equality_constraints: LinearEqualityConstraints
    linear_inequality_constraints: LinearInequalityConstraints
    nof_variables: int
    socp_constraints: SocpConstraintsWithoutConstants

    def __init__(self, nof_variables: int):
        self.nof_variables = nof_variables

    def _box_to_linear_constraints(self, jabr: ModelJabr):

        # initialize
        nof_box_constraints = (jabr.variable_sets[VariableType.PG].size +
                               jabr.variable_sets[VariableType.QG].size +
                               jabr.variable_sets[VariableType.CJJ].size)

        # upper bounds
        ub_matrix = sparse.csr_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        ub_matrix.setdiag(1)
        ub_vector = np.concatenate((jabr.box_constraint_sets[VariableType.PG].upper_bounds,
                                    jabr.box_constraint_sets[VariableType.QG].upper_bounds,
                                    jabr.box_constraint_sets[VariableType.CJJ].upper_bounds))
        ub_mask = np.invert(np.isnan(ub_vector))

        # lower bounds
        lb_matrix = sparse.csr_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        lb_matrix.setdiag(-1)
        lb_vector = np.concatenate((jabr.box_constraint_sets[VariableType.PG].lower_bounds,
                                    jabr.box_constraint_sets[VariableType.QG].lower_bounds,
                                    jabr.box_constraint_sets[VariableType.CJJ].lower_bounds))
        lb_mask = np.invert(np.isnan(lb_vector))

        # combine
        matrix = sparse.vstack((ub_matrix[ub_mask, :],
                                lb_matrix[lb_mask, :]), 'csr')
        vector = np.concatenate((ub_vector[ub_mask], -lb_vector[lb_mask]))
        self.linear_inequality_constraints = LinearInequalityConstraints(matrix, vector)

        # equalities
        eq_matrix = sparse.csr_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        eq_matrix.setdiag(1)
        eq_vector = np.concatenate((jabr.box_constraint_sets[VariableType.PG].equalities,
                                    jabr.box_constraint_sets[VariableType.QG].equalities,
                                    jabr.box_constraint_sets[VariableType.CJJ].equalities))
        eq_mask = np.invert(np.isnan(eq_vector))
        if not eq_mask.any():
            return
        new_equality_constraints = LinearEqualityConstraints(eq_matrix[eq_mask], eq_vector[eq_mask])
        self.linear_equality_constraints = (LinearEqualityConstraints
                                            .combine_linear_equality_constraints([new_equality_constraints,
                                                                                  self.linear_equality_constraints]))


    @classmethod
    def from_jabr(cls, jabr: ModelJabr):

        # initialize
        socp = cls(jabr.nof_variables)

        # initial values
        socp.initial_values = jabr.initial_values

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
