import numpy as np


class Lines:
    buses_from: np.ndarray[int]
    buses_to: np.ndarray[int]
    max_apparent_powers: np.ndarray[float]
    nof_lines: int

    def __init__(self, buses_from: np.ndarray[int], buses_to: np.ndarray[int], max_apparent_powers: np.ndarray[float]):
        self.nof_lines = max_apparent_powers.size
        assert buses_from.size == self.nof_lines
        assert buses_to.size == self.nof_lines
        self.buses_from = buses_from
        self.buses_to = buses_to
        self.max_apparent_powers = max_apparent_powers
