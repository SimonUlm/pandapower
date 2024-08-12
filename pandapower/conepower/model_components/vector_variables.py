import numpy as np

from pandapower.conepower.types.variable_type import VariableType


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
