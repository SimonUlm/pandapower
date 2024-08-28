import numpy as np


class GeneratorCost:
    nof_generators: int
    linear_coefficients: np.ndarray
    quadratic_coefficients: np.ndarray

    def __init__(self,
                 linear_coefficients: np.ndarray,
                 quadratic_coefficients: np.ndarray = None):
        # initialize
        self.nof_generators = linear_coefficients.size

        # linear coefficients
        self.linear_coefficients = linear_coefficients

        # quadratic coefficients
        if quadratic_coefficients is not None:
            assert quadratic_coefficients.size == self.nof_generators
            self.quadratic_coefficients = quadratic_coefficients
        else:
            self.quadratic_coefficients = np.zeros(self.nof_generators, dtype=float)

    def is_linear(self):
        return np.all(self.quadratic_coefficients == 0)
