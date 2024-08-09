import math
from typing import Dict

import scipy.sparse as sparse
import numpy as np

from pandapower.convexpower.models.model_components import VariableSet
from pandapower.convexpower.models.model_jabr import ModelJabr
from pandapower.convexpower.types.variable_type import VariableType


def _recover_angle_at_bus(jabr: ModelJabr,
                          new_angles: VariableSet,
                          connectivity_matrix: sparse.csc_matrix,
                          index: int,
                          index_from: int = -1):

    # update
    if index_from == -1:
        new_angles.initial_values[index] = 0
    else:
        index_cjk_sjk = connectivity_matrix[index_from, index]
        angle = new_angles.initial_values[index_from] + math.atan2(jabr.variable_sets[VariableType.SJK]
                                                                   .initial_values[index_cjk_sjk],
                                                                   jabr.variable_sets[VariableType.CJK]
                                                                   .initial_values[index_cjk_sjk])
        # TODO: Ich berechne aus versehen W^T anstatt W. Das liegt daran, dass ich jede Zeile der Amittanzmatrix mit den Spalten anstatt den Zeilen von W multipliziere.
        #  Schneller Fix: + statt - bei Winkelberechnung.
        new_angles.initial_values[index] = angle

    # block the way back
    col_start = connectivity_matrix.indptr[index]
    col_end = connectivity_matrix.indptr[index + 1]
    connectivity_matrix.data[col_start:col_end] = -1

    # repeat for all neighbors
    row = connectivity_matrix.getrow(index)
    mask = row.data != -1
    indices_to: np.ndarray = row.indices[mask]
    for idx in indices_to:
        _recover_angle_at_bus(jabr, new_angles, connectivity_matrix, idx, index)
    pass


def _recover_angles_from_radial_network(jabr: ModelJabr, variable_sets: Dict[VariableType, VariableSet]):

    # initialize
    connectivity_matrix: sparse.csc_matrix = jabr.cjk_indices_matrix.tocsc(copy=True)
    connectivity_matrix.setdiag(0)  # temp
    connectivity_matrix.eliminate_zeros()  # temp
    offset = (jabr.variable_sets[VariableType.PG].size +
              jabr.variable_sets[VariableType.QG].size +
              jabr.variable_sets[VariableType.CJJ].size)
    connectivity_matrix.data -= offset

    # start with bus 0
    _recover_angle_at_bus(jabr, variable_sets[VariableType.UANG], connectivity_matrix, 0)


def jabr_to_opf(jabr: ModelJabr) -> (Dict[VariableType, VariableSet], np.ndarray):

    # initialize
    nof_variables = (jabr.variable_sets[VariableType.PG].size +
                     jabr.variable_sets[VariableType.QG].size +
                     jabr.variable_sets[VariableType.CJJ].size * 2)
    variables = np.empty(nof_variables)
    variable_sets = {}

    # TODO: Fix horrible code below
    # pg
    starting_index = 0
    ending_index = jabr.variable_sets[VariableType.PG].size
    variable_sets[VariableType.PG] = VariableSet(VariableType.PG,
                                                 jabr.variable_sets[VariableType.PG].size,
                                                 jabr.variable_sets[VariableType.PG].initial_values,
                                                 variables[starting_index:ending_index])

    # qg
    starting_index = ending_index
    ending_index = starting_index + jabr.variable_sets[VariableType.QG].size
    variable_sets[VariableType.QG] = VariableSet(VariableType.QG,
                                                 jabr.variable_sets[VariableType.QG].size,
                                                 jabr.variable_sets[VariableType.QG].initial_values,
                                                 variables[starting_index:ending_index])

    # umag
    starting_index = ending_index
    ending_index = starting_index + jabr.variable_sets[VariableType.CJJ].size
    variable_sets[VariableType.UMAG] = VariableSet(VariableType.UMAG,
                                                   jabr.variable_sets[VariableType.CJJ].size,
                                                   np.sqrt(jabr.variable_sets[VariableType.CJJ].initial_values),
                                                   variables[starting_index:ending_index])

    # uang TODO: Only works for bus 0 as the ref bus with angle 0!!!
    starting_index = ending_index
    ending_index = starting_index + jabr.variable_sets[VariableType.CJJ].size
    variable_sets[VariableType.UANG] = VariableSet(VariableType.UANG,
                                                   jabr.variable_sets[VariableType.CJJ].size,
                                                   np.empty(jabr.variable_sets[VariableType.CJJ].size),
                                                   variables[starting_index:ending_index])
    _recover_angles_from_radial_network(jabr, variable_sets)

    return variable_sets, variables
