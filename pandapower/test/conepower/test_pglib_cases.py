import os
from math import log10, floor

import pytest

import pandapower as pp


LOG_OUT = os.path.join(pp.pp_dir, 'test', 'conepower', 'logs', 'validation.log')
FOLDER = os.path.join(pp.pp_dir, 'test', 'conepower', 'testfiles')
REL_ERROR_TOL = 1e-6


def assert_is_close_to_pm_result(filename, pm_result: float):
    file = os.path.join(FOLDER, filename)
    net = pp.converter.from_mpc(file)
    pp.runconvopp(net,
                  calculate_voltage_angles=True,
                  check_connectivity=True,
                  suppress_warnings=False,
                  relaxation='jabr',
                  enforce_ext_grid_vm=False,
                  flow_limit='S')
    rel_error = abs(net['res_cost'] - pm_result) / abs(pm_result)
    with open(LOG_OUT, 'a') as log_file:
        log_file.write(filename + ": " + str(rel_error) + "\n")
    assert rel_error < REL_ERROR_TOL


def test_pglib_opf_case3_lmbd():
    filename = "pglib_opf_case3_lmbd.m"
    pm_soc = 5.7361736882482710e+03
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  5.8126429350361796e+03
    # soc: 5.7361736882482710e+03


def test_pglib_opf_case5_pjm():
    filename = "pglib_opf_case5_pjm.m"
    pm_soc = 1.4999714987172934e+04
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  1.7551890838594758e+04
    # soc: 1.4999714987172934e+04


def test_pglib_opf_case14_ieee():
    filename = "pglib_opf_case14_ieee.m"
    pm_soc = 2.1757042522644620e+03
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  2.1780804108196430e+03
    # soc: 2.1757042522644620e+03


def test_pglib_opf_case24_ieee_rts():
    filename = "pglib_opf_case24_ieee_rts.m"
    pm_soc = 6.3344571423532638e+04
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  6.3352201086064109e+04
    # soc: 6.3344571423532638e+04


def test_pglib_opf_case30_as():
    filename = "pglib_opf_case30_as.m"
    pm_soc = 8.0267981967350920e+02
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  8.0312731038529728e+02
    # soc: 8.0267981967350920e+02


def test_pglib_opf_case30_ieee():
    filename = "pglib_opf_case30_ieee.m"
    pm_soc = 6.6621535198590464e+03
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  8.2085154279445469e+03
    # soc: 6.6621535198590464e+03


def test_pglib_opf_case39_epri():
    filename = "pglib_opf_case39_epri.m"
    pm_soc = 1.3765405922175694e+05
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  1.3841556252877225e+05
    # soc: 1.3765405922175694e+05


def test_pglib_opf_case57_ieee():
    filename = "pglib_opf_case57_ieee.m"
    pm_soc = 3.7529705912985381e+04
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  3.7589338204193133e+04
    # soc: 3.7529705912985381e+04


def test_pglib_opf_case73_ieee_rts():
    filename = "pglib_opf_case73_ieee_rts.m"
    pm_soc = 1.8970605846975342e+05
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  1.8976407716737359e+05
    # soc: 1.8970605846975342e+05


def test_pglib_opf_case89_pegase():
    filename = "pglib_opf_case89_pegase.m"
    pm_soc = 1.0648170198054370e+05
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  1.0728567306998365e+05
    # soc: 1.0648170198054370e+05


def test_pglib_opf_case118_ieee():
    filename = "pglib_opf_case118_ieee.m"
    pm_soc = 9.6335840619467548e+04
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  9.7213606938959303e+04
    # soc: 9.6335840619467548e+04


def test_pglib_opf_case162_ieee_dtc():
    filename = "pglib_opf_case162_ieee_dtc.m"
    pm_soc = 1.0165491193691417e+05
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  1.0807564369843053e+05
    # soc: 1.0165491193691417e+05


def test_pglib_opf_case179_goc():
    filename = "pglib_opf_case179_goc.m"
    pm_soc = 7.5309270724975877e+05
    assert_is_close_to_pm_result(filename, pm_soc)
    # ac:  7.5426641416841024e+05
    # soc: 7.5309270724975877e+05
