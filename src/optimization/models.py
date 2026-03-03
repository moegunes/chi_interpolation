import numpy as np

from utils.physics import J_n_m_kFr, delta_C


def delta_pi(r, rs, params, get_constraints=False):
    """
    Two-mode model with r^2 prefactor:
        ∆chi(r) = [ B0 e^{-\alpha_0 r} cos(k0 r + \phi_0)
                   + B1 e^{-\alpha_1 r} cos(k1 r + \phi_1) ]

    Parameters (fit):
        params = [\alpha_0, f0, \phi_0, \alpha_1, f1, \phi_1]
    """
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    kFr = np.asarray(kF * r, float)

    alpha0, f0, phi0, alpha1, f1, phi1 = params
    # phi0, phi1 = np.mod(phi0, 2 * np.pi), np.mod(phi1, 2 * np.pi)
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1

    # Coeffs
    J0 = J_n_m_kFr(0, k0, alpha0, phi0, kF)
    J1 = J_n_m_kFr(0, k1, alpha1, phi1, kF)
    J3 = J_n_m_kFr(1, k0, alpha0, phi0, kF)
    J4 = J_n_m_kFr(1, k1, alpha1, phi1, kF)

    b = np.array([delta_C(1, rs), delta_C(0, rs)])
    # Mmat = np.array([[c0, c1],
    #                 [J0, J1]])
    Mmat = np.array([[J3, J4], [J0, J1]])
    # Solve for B0, B1
    B0, B1 = np.linalg.solve(Mmat, b)

    if get_constraints:
        return B0, B1
    else:
        delta_chi = B0 * np.exp(-alpha0 * kFr) * np.cos(k0 * kFr + phi0) + B1 * np.exp(
            -alpha1 * kFr
        ) * np.cos(k1 * kFr + phi1)
        return delta_chi
