import scipy.sparse as sparse
import numpy as np

from pandapower.convexpower.model_components import *
from pandapower.convexpower.model_jabr import ModelJabr
from pandapower.convexpower.variable_type import VariableType


class ModelSocp:
    initial_values: np.ndarray
    linear_cost: np.ndarray
    linear_equality_constraints: LinearEqualityConstraints
    linear_inequality_constraints: LinearInequalityConstraints
    nof_variables: int
    socp_constraints: SocpConstraintsWithoutConstants

    def __init__(self, nof_variables: int):
        self.nof_variables = nof_variables

    def _box_to_linear_inequality_constraints(self, jabr: ModelJabr):

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

        # lower bounds
        lb_matrix = sparse.csr_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        lb_matrix.setdiag(-1)
        lb_vector = np.concatenate((jabr.box_constraint_sets[VariableType.PG].lower_bounds,
                                    jabr.box_constraint_sets[VariableType.QG].lower_bounds,
                                    jabr.box_constraint_sets[VariableType.CJJ].lower_bounds))

        # combine
        matrix = sparse.vstack((ub_matrix,
                                lb_matrix), 'csr')
        vector = np.concatenate((ub_vector, lb_vector))
        self.linear_inequality_constraints = LinearInequalityConstraints(matrix, vector)

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
        socp._box_to_linear_inequality_constraints(jabr)

        # socp constraints
        socp.socp_constraints = jabr.socp_constraints

        return socp
