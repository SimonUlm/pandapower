from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import sparse


class Constraints(ABC):
    nof_constraints: int = 0

    def __init__(self, nof_constraints):
        self.nof_constraints = nof_constraints

    @abstractmethod
    def __add__(self, other):
        pass

    def __iadd__(self, other):
        return self + other

    @abstractmethod
    def prepend_variable(self):
        pass

    @abstractmethod
    def scaled(self):
        pass

    @abstractmethod
    def to_cone_formulation(self) -> Tuple[sparse.coo_matrix, np.ndarray, int]:
        pass

    def is_empty(self) -> bool:
        return self.nof_constraints == 0
