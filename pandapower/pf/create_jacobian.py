from numpy import complex128, float64, int32, ones
from numpy.core.multiarray import zeros, empty, array
from scipy.sparse import csr_matrix as sparse, vstack, hstack

from pandapower.pypower.dSbus_dV import dSbus_dV

try:
    # numba functions
    from pandapower.pf.create_jacobian_numba import create_J, create_J2
    from pandapower.pf.dSbus_dV_numba import dSbus_dV_numba_sparse
except ImportError:
    pass


def _create_J_with_numba(Ybus, V, pvpq, pq, createJ, pvpq_lookup, npv, npq):
    Ibus = zeros(len(V), dtype=complex128)
    # create Jacobian from fast calc of dS_dV
    dVm_x, dVa_x = dSbus_dV_numba_sparse(Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V), Ibus)

    # data in J, space preallocated is bigger than acutal Jx -> will be reduced later on
    Jx = empty(len(dVm_x) * 4, dtype=float64)
    # row pointer, dimension = pvpq.shape[0] + pq.shape[0] + 1
    Jp = zeros(pvpq.shape[0] + pq.shape[0] + 1, dtype=int32)
    # indices, same with the preallocated space (see Jx)
    Jj = empty(len(dVm_x) * 4, dtype=int32)

    # fill Jx, Jj and Jp
    createJ(dVm_x, dVa_x, Ybus.indptr, Ybus.indices, pvpq_lookup, pvpq, pq, Jx, Jj, Jp)

    # resize before generating the scipy sparse matrix
    Jx.resize(Jp[-1], refcheck=False)
    Jj.resize(Jp[-1], refcheck=False)

    # generate scipy sparse matrix
    dimJ = npv + npq + npq
    J = sparse((Jx, Jj, Jp), shape=(dimJ, dimJ))

    return J


def _create_J_without_numba(Ybus, V, pvpq, pq, contribution_factors, dist_slack):
    # create Jacobian with standard pypower implementation.
    if dist_slack:
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V[:-1])
    else:
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

    ## evaluate Jacobian
    J11 = dS_dVa[array([pvpq]).T, pvpq].real
    J12 = dS_dVm[array([pvpq]).T, pq].real
    if len(pq) > 0 or dist_slack:
        J21 = dS_dVa[array([pq]).T, pvpq].imag
        J22 = dS_dVm[array([pq]).T, pq].imag
        if dist_slack:
            J13 = contribution_factors[1:].reshape(-1,1)
            J23 = ones(len(pq)).reshape(-1,1)
            print('___J11:')
            print(J11.toarray())
            print('___J12:')
            print(J12.toarray())
            print('___J13:')
            print(J13)
            print('___J11J12J13')
            print(hstack([J11, J12, J13]).toarray())
            print('___J21:')
            print(J21.toarray())
            print('___J22:')
            print(J22.toarray())
            print('___J23:')
            print(J23)
            print('___J21J22J23')
            print(hstack([J21, J22, J23]).toarray())
            J = vstack([
                hstack([J11, J12, J13]),
                hstack([J21, J22, J23])
            ], format="csr")
        else:
            J = vstack([
                hstack([J11, J12]),
                hstack([J21, J22])
            ], format="csr")
    else:
        J = vstack([
            hstack([J11, J12])
        ], format="csr")
    return J


def create_jacobian_matrix(Ybus, V, pvpq, pq, createJ, pvpq_lookup, npv, npq, numba, contribution_factors, dist_slack):
    if numba and not dist_slack:
        J = _create_J_with_numba(Ybus, V, pvpq, pq, createJ, pvpq_lookup, npv, npq)
    else:
        J = _create_J_without_numba(Ybus, V, pvpq, pq, contribution_factors, dist_slack)
    return J


def get_fastest_jacobian_function(pvpq, pq, numba, dist_slack):
    if numba and not dist_slack:
        if len(pvpq) == len(pq):
            create_jacobian = create_J2
        else:
            create_jacobian = create_J
    else:
        create_jacobian = None
    return create_jacobian
