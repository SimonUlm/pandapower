# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves the power flow using a full Newton's method.
"""

from numpy import angle, exp, linalg, conj, r_, Inf, arange, zeros, max, zeros_like, column_stack, append
from numpy import flatnonzero as find
from scipy.sparse.linalg import spsolve

from pandapower.pf.iwamoto_multiplier import _iwamoto_step
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.create_jacobian import create_jacobian_matrix, get_fastest_jacobian_function
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, CON_FAC
from pandapower.pypower.idx_bus import BUS_I, CON_FAC


def newtonpf(Ybus, Sbus, V0, pv, pq, ppci, options):
    """Solves the power flow using a full Newton's method.
    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles.
    @see: L{runpf}
    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    Modified by University of Kassel (Florian Schaefer) to use numba
    """

    # options
    tol = options['tolerance_mva']
    max_it = options["max_iteration"]
    numba = options["numba"]
    iwamoto = options["algorithm"] == "iwamoto_nr"
    voltage_depend_loads = options["voltage_depend_loads"]
    dist_slack = options["distributed_slack"]
    v_debug = options["v_debug"]
    use_umfpack = options["use_umfpack"]
    permc_spec = options["permc_spec"]

    baseMVA = ppci['baseMVA']
    bus = ppci['bus']
    gen = ppci['gen']
    contribution_factors = bus[:, CON_FAC]  ## contribution factors for distributed slack

    # initialize
    i = 0
    V = V0
    if dist_slack:
        V = append(V, 0)
    Va = angle(V)
    Vm = abs(V)
    dVa, dVm = None, None
    if iwamoto:
        dVm, dVa = zeros_like(Vm), zeros_like(Va)

    if v_debug:
        Vm_it = Vm.copy()
        Va_it = Va.copy()
    else:
        Vm_it = None
        Va_it = None

    # set up indexing for updating V
    pvpq = r_[pv, pq]
    # generate lookup pvpq -> index pvpq (used in createJ)
    pvpq_lookup = zeros(max(Ybus.indices) + 1, dtype=int)
    pvpq_lookup[pvpq] = arange(len(pvpq))

    # get jacobian function
    createJ = get_fastest_jacobian_function(pvpq, pq, numba, dist_slack)

    npv = len(pv)
    npq = len(pq)
    j1 = 0
    j2 = npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses

    # evaluate F(x0)
    F = _evaluate_Fx(Ybus, V, Sbus, pv, pq, contribution_factors, dist_slack)
    converged = _check_for_convergence(F, tol)

    Ybus = Ybus.tocsr()
    J = None

    # do Newton iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        J = create_jacobian_matrix(Ybus, V, pvpq, pq, createJ, pvpq_lookup, npv, npq, numba, contribution_factors, dist_slack)

        dx = -1 * spsolve(J, F, permc_spec=permc_spec, use_umfpack=use_umfpack)
        # update voltage
        if npv and not iwamoto:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq and not iwamoto:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]

        # iwamoto multiplier to increase convergence
        if iwamoto:
            Vm, Va = _iwamoto_step(Ybus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6)

        V = Vm * exp(1j * Va)
        Vm = abs(V)  # update Vm and Va again in case
        Va = angle(V)  # we wrapped around with a negative Vm

        if v_debug:
            Vm_it = column_stack((Vm_it, Vm))
            Va_it = column_stack((Va_it, Va))

        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=Vm)

        F = _evaluate_Fx(Ybus, V, Sbus, pv, pq, contribution_factors, dist_slack)

        converged = _check_for_convergence(F, tol)

    return V, converged, i, J, Vm_it, Va_it


def _evaluate_Fx(Ybus, V, Sbus, pv, pq, contribution_factors, dist_slack):
    # evalute F(x)
    if dist_slack:
        # we smuggle the guess for the overall slack mismatch as the last element of V
        print(V)
        print(contribution_factors)
        mis = V[:-1] * conj(Ybus * V[:-1]) - Sbus + contribution_factors * V[-1]
    else:
        mis = V * conj(Ybus * V) - Sbus
    F = r_[mis[pv].real,
           mis[pq].real,
           mis[pq].imag]
    return F


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, Inf) < tol
