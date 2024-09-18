from typing import Dict
from sys import stdout

import numpy as np

from pandapower.conepower.model_components.vector_variable import VariableSet
from pandapower.conepower.types.variable_type import VariableType
from pandapower.conepower.unit_conversions.per_unit_converter import PerUnitConverter
from pandapower.pypower.idx_bus import VM, VA
from pandapower.pypower.idx_gen import PG, QG, VG, GEN_BUS
from pandapower.pypower.opf_model import opf_model


def postprocess(ppc: Dict,
                om: opf_model,
                elapsed_time: float,
                success: bool,
                objective_value: float,
                variables: np.ndarray[float],
                variables_sets: Dict[VariableType, VariableSet]) -> Dict:
    # initialize
    result = ppc
    result["om"] = om
    result["et"] = elapsed_time
    result["success"] = success
    result["raw"] = None
    result["f"] = objective_value
    result["x"] = variables
    result["mu"] = None

    # extract relevant data
    base_mva, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    # create converter to convert from p.u. to respective SI units
    converter = PerUnitConverter(base_mva)

    # update buses
    bus[:, VM] = variables_sets[VariableType.UMAG].values
    # bus[:, VA] = np.degrees(variables_sets[VariableType.UANG].values)  # TODO: This should be done here, not earlier
    bus[:, VA] = variables_sets[VariableType.UANG].values
    # TODO: Lagrange multiplier of box constraints MU_VMAX, MU_VMIN and of power flow equation LAM_P, LAM_Q

    # update branch
    # TODO: Branch flows PF, QF, PT, QT and multipliers MU_SF, MU_ST

    # update gen
    gen[:, PG] = converter.to_power(variables_sets[VariableType.PG].values)  # TODO: Fix typing
    gen[:, QG] = converter.to_power(variables_sets[VariableType.QG].values)
    gen[:, VG] = variables_sets[VariableType.UMAG].values[gen[:, GEN_BUS].astype(int)]
    # TODO: Lagrange multiplier of box constraints MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN

    return result
