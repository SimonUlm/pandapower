import numpy as np


class PerUnitConverter:
    _ref_power_mva: float
    _ref_voltage_kv: float

    def __init__(self, ref_power_mva: float, ref_voltage_kv: float = None):
        self._ref_power_mva = ref_power_mva
        self._ref_voltage_kv = ref_voltage_kv

    def from_linear_generator_cost(self, cost_per_mw: np.ndarray[float]) -> np.ndarray[float]:
        return cost_per_mw * self._ref_power_mva

    def from_power(self, power_mva: np.ndarray[float]) -> np.ndarray[float]:
        return power_mva / self._ref_power_mva

    def to_power(self, power_pu: np.ndarray[float]) -> np.ndarray[float]:
        return power_pu * self._ref_power_mva
