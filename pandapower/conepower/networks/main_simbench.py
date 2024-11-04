from copy import deepcopy
from os import listdir

import pandas as pd

import pandapower as pp
from pandapower.conepower.networks.process_simbench import StudyCase, InverterControlMode, lv_grid_to_pf, lv_grid_to_opf


NETWORK_PATH = "../networks"
EXACTNESS_OUTPUT_FILE_1 = "../../../../../Documents/Thesis/tables/exactness_1.tex"  # TODO: Do this properly
EXACTNESS_OUTPUT_FILE_2 = "../../../../../Documents/Thesis/tables/exactness_2.tex"


def format_float(x: float) -> str:
    if isinstance(x, float):
        return f"{x: .2e}"
    return str(x)


def format_names(df: pd.DataFrame) -> pd.DataFrame:
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

        # exactness
        if True:
            # conventional opf
            try:
                net = lv_grid_to_opf(pp.from_json(file), case)
                opf_net = deepcopy(net)
                pp.runopp(opf_net,
                          verbose=False,
                          calculate_voltage_angles=False,
                          PDIPM_FEASTOL=1e-9,
                          PDIPM_GRADTOL=1e-9,
                          PDIPM_COMPTOL=1e-9,
                          PDIPM_COSTTOL=1e-9)
                res_opf = opf_net['res_cost']
                # relaxed opf
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
                perc = abs((res_opf - res_conv) / res_opf)
                relaxation_errors[-1].append(output['relaxation_error'])
                optimality_gaps[-1].append(perc)
                if res_opf < res_conv:
                    print(str(perc) + " (res_opf < res_conv)")
                else:
                    print(perc)
            except Exception:
                print("One of the solvers did not converge!")
                relaxation_errors[-1].append(float('nan'))
                optimality_gaps[-1].append(float('nan'))

        if False:
            try:
                # pf
                pf_net = lv_grid_to_pf(pp.from_json(file),
                                       case,
                                       InverterControlMode.CONST_PHI)
                pp.runpp(pf_net,
                         calculate_voltage_angles=False,
                         init='flat',
                         tolerance_mva=1e-9,
                         trafo_model='pi',
                         voltage_depend_loads=False)
                res_pf = pf_net['res_ext_grid']['p_mw'][0]
                # relaxed opf
                cone_net = lv_grid_to_opf(pp.from_json(file), case)
                pp.runconvopp(cone_net,
                              verbose=False,
                              calculate_voltage_angles=False,
                              suppress_warnings=False,
                              enforce_ext_grid_vm=True,
                              flow_limit='I')
                res_conv = cone_net['res_cost']
                # print("PF  = " + str(res_pf))
                # print("OPF = " + str(res_conv))
                print("Difference [kW] = " + str((res_pf - res_conv) * 1000))
                pass
            except Exception:
                print("One of the solvers did not converge!")

# postprocess output
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
    latex_table_1 = format_names(frame_1.map(format_float)).to_latex(index=False)
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
    latex_table_2 = format_names(frame_2.map(format_float)).to_latex(index=False)
    with open(EXACTNESS_OUTPUT_FILE_2, 'w') as file:
        file.write(latex_table_2)
