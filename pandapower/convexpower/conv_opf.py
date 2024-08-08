from time import perf_counter

from numpy import zeros, c_, shape
from pandapower.pypower.idx_brch import MU_ANGMAX
from pandapower.pypower.idx_bus import MU_VMIN
from pandapower.pypower.idx_gen import MU_QMIN

from pandapower.pypower.opf_args import opf_args2
from pandapower.pypower.opf_execute import opf_execute
from pandapower.pypower.opf_setup import opf_setup

from pandapower.convexpower.models.model_jabr import ModelJabr
from pandapower.convexpower.models.model_opf import ModelOpf
from pandapower.convexpower.models.model_socp import ModelSocp
from pandapower.convexpower.types.relaxation_type import RelaxationType


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

    # apply relaxation
    if relaxation_type is RelaxationType.JABR:
        jabr = ModelJabr.from_model_opf(model)
        socp = ModelSocp.from_jabr(jabr)
    else:
        assert False

    # execute the OPF
    results, success, raw = opf_execute(om, ppopt)

    # finish preparing output
    et = perf_counter() - t0

    results['et'] = et
    results['success'] = success
    results['raw'] = raw

    return results
