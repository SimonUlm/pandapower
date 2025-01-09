import math
from typing import Tuple
from os import listdir

import numpy as np
import pandas as pd

import pandapower as pp
from pandapower.conepower.study.process_simbench import (InverterControlMode, StudyCase,
                                                         lv_grid_to_opf, lv_grid_to_pf)


NETWORK_PATH = "../networks"
OUTPUT_FILE_1 = "../../../../../Documents/Thesis/tables/power_flow_1.tex"  # TODO: Do this properly
OUTPUT_FILE_2 = "../../../../../Documents/Thesis/tables/power_flow_2.tex"


def _format_float(x: float) -> str:
    if isinstance(x, float):
        return str(round(x, 2))
    return str(x)


def _format_names(df: pd.DataFrame) -> pd.DataFrame:
    formatted_names = []
    previous_name = None
    for name in df['Grid']:
        if name == previous_name:
            formatted_names.append("")
        else:
            formatted_names.append(name)
        previous_name = name
    df['Grid'] = formatted_names
    return df


def _check_bounds(net: pp.pandapowerNet) -> Tuple[bool, bool, bool, bool]:
    # check for voltage bounds
    is_min_vm_valid = np.all(net.res_bus['vm_pu'] > net.bus['min_vm_pu'])
    is_max_vm_valid = np.all(net.res_bus['vm_pu'] < net.bus['max_vm_pu'])

    # check for line constraints
    is_max_line_valid = np.all(net.res_line['loading_percent'] < net.line['max_loading_percent'])
    is_max_trafo_valid = np.all(net.res_trafo['loading_percent'] < net.trafo['max_loading_percent'])

    return is_min_vm_valid, is_max_vm_valid, is_max_line_valid, is_max_trafo_valid


files = [file for file in listdir(NETWORK_PATH) if "json" in file]
files.sort()
study_cases = [StudyCase.HIGH_LOAD_HIGH_PV,
               StudyCase.LOW_LOAD_HIGH_PV,
               StudyCase.HIGH_LOAD_LOW_PV,
               StudyCase.LOW_LOAD_LOW_PV]

# prepare output
names = [file.split("--")[0] for file in files]
year_indices = [int(file.split("--")[1][0]) for file in files]
years = [(2024 if idx == 1 else (2034 if idx == 2 else 2016)) for idx in year_indices]
absolutes = []
percentages = []

for case in study_cases:
    # prepare output
    absolutes.append([])
    percentages.append([])

    for file in files:
        print(file + ": " + case.to_str())
        success = True

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
            success = False

        # pf
        pf_net = lv_grid_to_pf(pp.from_json(NETWORK_PATH + '/' + file),
                               case,
                               InverterControlMode.PF_95)
        pp.runpp(pf_net,
                 calculate_voltage_angles=False,
                 init='flat',
                 tolerance_mva=1e-9,
                 trafo_model='pi',
                 voltage_depend_loads=False)
        is_min_vm_valid, is_max_vm_valid, is_max_line_valid, is_max_trafo_valid = _check_bounds(pf_net)
        is_valid = is_min_vm_valid and is_max_vm_valid and is_max_line_valid and is_max_trafo_valid

        # compare
        absolute = float('nan')
        percentage = float('nan')
        if success and is_valid:
            res_pf = pf_net['res_ext_grid']['p_mw'][0]
            absolute = 1000 * (res_pf - res_conv)
            percentage = 100 * (res_pf - res_conv) / abs(res_pf)
            print("Difference = " + str(round(absolute, 2)) + "kW or " + str(round(percentage, 2)) + "%")
        elif not success:
            print("Jabr's relaxation did not converge!")
        else:
            if not is_min_vm_valid:
                print("Below minimum voltage")
            if not is_max_vm_valid:
                print("Above maximum voltage")
            if not is_max_line_valid:
                print("Above maximum line current")
            if not is_max_trafo_valid:
                print("Above maximum trafo current")
            absolute = 0.
            percentage = 100.
        absolutes[-1].append(absolute)
        percentages[-1].append(percentage)

# print to latex
if len(absolutes) > 0 and len(percentages) > 0:
    nof_cases = len(study_cases)
    inter_keys = sum([['Abs [kW] (' + case.to_str() + ')', 'Perc (' + case.to_str() + ')'] for case in study_cases], [])
    inter_values = [value for pair in zip(absolutes, percentages) for value in pair]
    inter_len = len(inter_keys)
    assert len(inter_values) == inter_len
    assert inter_len % 2 == 0
    # table 1
    keys_1 = (['Grid']
              + ['Case']
              + inter_keys[:inter_len // 2])
    values_1 = ([names, years]
                + inter_values[:inter_len // 2])
    data_1 = dict(zip(keys_1, values_1))
    frame_1 = pd.DataFrame(data_1)
    latex_table_1 = _format_names(frame_1.map(_format_float)).to_latex(index=False)
    with open(OUTPUT_FILE_1, 'w') as file:
        file.write(latex_table_1)
    # table 2
    keys_2 = (['Grid']
              + ['Case']
              + inter_keys[inter_len // 2:])
    values_2 = ([names, years]
                + inter_values[inter_len // 2:])
    data_2 = dict(zip(keys_2, values_2))
    frame_2 = pd.DataFrame(data_2)
    latex_table_2 = _format_names(frame_2.map(_format_float)).to_latex(index=False)
    with open(OUTPUT_FILE_2, 'w') as file:
        file.write(latex_table_2)
