from abc import ABC, abstractmethod


class Constraints(ABC):
    nof_constraints: int

    def __init__(self, nof_constraints):
        self.nof_constraints = nof_constraints

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __iadd__(self, other):
        pass

    @abstractmethod
    def to_cone_formulation(self):
        pass
