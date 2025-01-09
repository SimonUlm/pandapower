import math
from copy import deepcopy
from os import listdir

import pandas as pd

import pandapower as pp
from pandapower.auxiliary import OPFNotConverged
from pandapower.conepower.study.process_simbench import StudyCase, lv_grid_to_opf


NETWORK_PATH = "../networks"
EXACTNESS_OUTPUT_FILE_1 = "../../../../../Documents/Thesis/tables/exactness_1.tex"  # TODO: Do this properly
EXACTNESS_OUTPUT_FILE_2 = "../../../../../Documents/Thesis/tables/exactness_2.tex"


def _format_float(x: float) -> str:
    if isinstance(x, float):
        return f"{x: .2e}"
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
relaxation_errors = []
optimality_gaps = []

for case in study_cases:
    # prepare output
    relaxation_errors.append([])
    optimality_gaps.append([])

    for file in files:
        print(file + ": " + case.to_str())
        net = lv_grid_to_opf(pp.from_json(NETWORK_PATH + '/' + file), case)

        # conventional opf
        try:
            opf_net = deepcopy(net)
            pp.runopp(opf_net,
                      verbose=False,
                      calculate_voltage_angles=False,
                      PDIPM_FEASTOL=1e-9,
                      PDIPM_GRADTOL=1e-9,
                      PDIPM_COMPTOL=1e-9,
                      PDIPM_COSTTOL=1e-9)
            res_opf = opf_net['res_cost']
        except OPFNotConverged:
            res_opf = float('nan')

        # relaxed opf
        try:
            cone_net = deepcopy(net)
            output = {}
            pp.runconvopp(cone_net,
                          verbose=False,
                          calculate_voltage_angles=False,
                          suppress_warnings=False,
                          enforce_ext_grid_vm=True,
                          flow_limit='I',
                          output=output)
            res_conv = cone_net['res_cost']
        except ValueError:
            res_conv = float('nan')

        # compare
        relaxation_error = float('nan')
        optimality_gap = float('nan')
        if not math.isnan(res_opf) and not math.isnan(res_conv):
            optimality_gap = abs((res_opf - res_conv) / res_opf)
            relaxation_error = output['relaxation_error']
            print(optimality_gap)
        elif math.isnan(res_opf) and math.isnan(res_conv):
            print("Both solvers did not converge!")
        else:
            assert False
        relaxation_errors[-1].append(relaxation_error)
        optimality_gaps[-1].append(optimality_gap)


# print to latex
if len(relaxation_errors) > 0 and len(optimality_gaps) > 0:
    nof_cases = len(study_cases)
    inter_keys = sum([['Error (' + case.to_str() + ')', 'Gap (' + case.to_str() + ')'] for case in study_cases], [])
    inter_values = [value for pair in zip(relaxation_errors, optimality_gaps) for value in pair]
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
    with open(EXACTNESS_OUTPUT_FILE_1, 'w') as file:
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
    with open(EXACTNESS_OUTPUT_FILE_2, 'w') as file:
        file.write(latex_table_2)
