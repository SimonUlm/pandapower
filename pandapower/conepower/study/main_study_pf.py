from os import listdir

import pandas as pd

import pandapower as pp
from pandapower.conepower.study.process_simbench import (InverterControlMode, StudyCase,
                                                         lv_grid_to_opf, lv_grid_to_pf)


NETWORK_PATH = "../networks"


files = [file for file in listdir(NETWORK_PATH) if "json" in file]
files.sort()
study_cases = [StudyCase.HIGH_LOAD_HIGH_PV,
               StudyCase.LOW_LOAD_HIGH_PV,
               StudyCase.HIGH_LOAD_LOW_PV,
               StudyCase.LOW_LOAD_LOW_PV]

for case in study_cases:
    for file in files:
        print(file + ": " + case.to_str())

        # relaxed opf
        try:
            cone_net = lv_grid_to_opf(pp.from_json(NETWORK_PATH + '/' + file), case)
            pp.runconvopp(cone_net,
                          verbose=False,
                          calculate_voltage_angles=False,
                          suppress_warnings=False,
                          enforce_ext_grid_vm=True,
                          flow_limit='I')
            res_conv = cone_net['res_cost']
        except ValueError:
            print("Jabr's relaxation did not converge!")
            continue

        # pf
        pf_net = lv_grid_to_pf(pp.from_json(NETWORK_PATH + '/' + file),
                               case,
                               InverterControlMode.ZERO_Q)
        pp.runpp(pf_net,
                 calculate_voltage_angles=False,
                 init='flat',
                 tolerance_mva=1e-9,
                 trafo_model='pi',
                 voltage_depend_loads=False)
        res_pf = pf_net['res_ext_grid']['p_mw'][0]
        perc = 100 * (round((res_pf - res_conv) / abs(res_pf), 4))
        print("Difference [kW] = " + str((res_pf - res_conv) * 1000) + " or " + str(perc))
