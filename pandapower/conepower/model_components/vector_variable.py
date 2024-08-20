import numpy as np


class VariableSet:
    lower_bounds: np.ndarray
    size: int
    upper_bounds: np.ndarray
    values: np.ndarray

    def __init__(self,
                 values_data: np.ndarray,
                 lower_bounds: np.ndarray = None,
                 upper_bounds: np.ndarray = None,
                 values_allocated_memory: np.ndarray = None):
        self.size = values_data.size
        if values_allocated_memory is None:
            self.values = values_data
        else:
            assert values_allocated_memory.size == self.size
            self.values = values_allocated_memory
            np.copyto(self.values, values_data)
        if lower_bounds is not None:
            assert lower_bounds.size == self.size
            self.lower_bounds = lower_bounds
        if upper_bounds is not None:
            assert upper_bounds.size == self.size
            self.upper_bounds = upper_bounds
    # TODO: Introduce offset to determine the offset within the allocated memory.
    #  In the model classes, rename initial_values into variables.
