from time import perf_counter

import numpy as np
from numpy import zeros, c_, shape

from pandapower.conepower.models.model_opf import ModelOpf
from pandapower.conepower.models.model_jabr import ModelJabr
from pandapower.conepower.models.model_socp import ModelSocp
from pandapower.conepower.postprocessing import postprocess
from pandapower.conepower.types.relaxation_type import RelaxationType
from pandapower.conepower.types.optimization_type import OptimizationType
from pandapower.conepower.types.variable_type import VariableType  # required for debugging

from pandapower.pypower.idx_brch import MU_ANGMAX
from pandapower.pypower.idx_bus import MU_VMIN
from pandapower.pypower.idx_gen import MU_QMIN
from pandapower.pypower.opf_args import opf_args2
from pandapower.pypower.opf_setup import opf_setup


def conv_opf(ppc, ppopt, relaxation_str):
    # initialize
    t0 = perf_counter()

    # process input arguments
    ppc, ppopt = opf_args2(ppc, ppopt)

    # add zero columns to bus, gen, branch for multipliers, etc. if needed
    nb = shape(ppc['bus'])[0]
    nl = shape(ppc['branch'])[0]
    ng = shape(ppc['gen'])[0]
    if shape(ppc['bus'])[1] < MU_VMIN + 1:
        ppc['bus'] = c_[ppc['bus'], zeros((nb, MU_VMIN + 1 - shape(ppc['bus'])[1]))]
    if shape(ppc['gen'])[1] < MU_QMIN + 1:
        ppc['gen'] = c_[ppc['gen'], zeros((ng, MU_QMIN + 1 - shape(ppc['gen'])[1]))]
    if shape(ppc['branch'])[1] < MU_ANGMAX + 1:
        ppc['branch'] = c_[ppc['branch'], zeros((nl, MU_ANGMAX + 1 - shape(ppc['branch'])[1]))]

    # construct OPF model object
    om = opf_setup(ppc, ppopt)

    # convert to own model
    model = ModelOpf.from_om(om)

    # get the type of convex relaxation
    relaxation_type = RelaxationType.from_str(relaxation_str)
    assert relaxation_type is not RelaxationType.UNKNOWN

    # initialize
    x = None
    opt_model = None
    opt_type = OptimizationType.UNKNOWN

    # apply relaxation
    if relaxation_type is RelaxationType.JABR:
        jabr = ModelJabr.from_opf(model)
        opt_model = ModelSocp.from_jabr(jabr)
        opt_type = OptimizationType.SOCP
    else:
        assert False

    # execute the relaxed OPF
    assert opt_model is not None
    if opt_type is OptimizationType.SOCP:
        success, objective, resulting_variables = opt_model.solve()
    else:
        assert False

    # finish preparing output
    et = perf_counter() - t0

    result = ppc

    # calculate error and recover solution
    if relaxation_type is RelaxationType.JABR:
        jabr.set_values(resulting_variables)
        error = jabr.calculate_jabr_infeasibility()
        print('SOCP infeasibility: ' + str(error))
        variable_sets, variables = jabr.to_opf_variables()
        #np.copyto(model.values, variables)  # TODO: Refactor
        result = postprocess(ppc=ppc,
                             om=om,
                             elapsed_time=et,
                             success=success,
                             objective_value=objective,
                             constant_costs=model.active_generator_cost.constants,
                             variables=variables,
                             variables_sets=variable_sets)
    else:
        assert False

    return result
