import numpy as np

from utils.physics import J_n_cos, J_n_m, J_n_sin, delta_C


def delta_chi_nl(r, rs, params, get_constraints=False):
    """
    Two-mode model with r^2 prefactor, phase eliminated:

        X(r) = r^(3-gamma) * [
            e^{-alpha0 r} (C0 cos(k0 r) + D0 sin(k0 r))
          + e^{-alpha1 r} (C1 cos(k1 r) + D1 sin(k1 r))
        ]

    Fit parameters:
        params = [alpha0, f0, alpha1, f1]

    Linear amplitudes (solved from constraints):
        [C0, D0, C1, D1]
    """
    r = np.asarray(r, float)

    alpha0, f0, alpha1, f1 = params

    alpha0, alpha1 = np.abs(alpha0), np.abs(alpha1)
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1

    # ---- exact constraints ----
    # example: n = 1 and n = 0 (same as your original)

    Mmat = np.array(
        [
            [
                J_n_cos(0, k0, alpha0),
                J_n_sin(0, k0, alpha0),
                J_n_cos(0, k1, alpha1),
                J_n_sin(0, k1, alpha1),
            ],
            [
                J_n_cos(1, k0, alpha0),
                J_n_sin(1, k0, alpha0),
                J_n_cos(1, k1, alpha1),
                J_n_sin(1, k1, alpha1),
            ],
            [
                J_n_cos(2, k0, alpha0),
                J_n_sin(2, k0, alpha0),
                J_n_cos(2, k1, alpha1),
                J_n_sin(2, k1, alpha1),
            ],
            [
                J_n_cos(3, k0, alpha0),
                J_n_sin(3, k0, alpha0),
                J_n_cos(3, k1, alpha1),
                J_n_sin(3, k1, alpha1),
            ],
        ]
    )

    b = np.array(
        [
            delta_C(0, rs),
            delta_C(1, rs),
            delta_C(2, rs),
            delta_C(3, rs),
        ]
    )

    # Solve for amplitudes (minimum-norm if underdetermined)
    # C0, D0, C1, D1 = np.linalg.lstsq(Mmat, b, rcond=None)[0]
    C0, D0, C1, D1 = np.linalg.solve(Mmat, b)

    if get_constraints:
        return C0, D0, C1, D1

    # ---- build X(r) ----
    X = np.exp(-alpha0 * r) * (C0 * np.cos(k0 * r) + D0 * np.sin(k0 * r)) + np.exp(
        -alpha1 * r
    ) * (C1 * np.cos(k1 * r) + D1 * np.sin(k1 * r))

    return X


def delta_chi(r, rs, params, get_constraints=False):
    """
    Two-mode model with r^2 prefactor:
        ∆chi(r) = [ B0 e^{-\alpha_0 r} cos(k0 r + \phi_0)
                   + B1 e^{-\alpha_1 r} cos(k1 r + \phi_1) ]

    Parameters (fit):
        params = [\alpha_0, f0, \phi_0, \alpha_1, f1, \phi_1]
    """
    r = np.asarray(r, float)
    alpha0, f0, phi0, alpha1, f1, phi1 = params
    # phi0, phi1 = np.mod(phi0, 2 * np.pi), np.mod(phi1, 2 * np.pi)
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1

    # Coeffs
    J0 = J_n_m(0, k0, alpha0, phi0)
    J1 = J_n_m(0, k1, alpha1, phi1)
    J3 = J_n_m(1, k0, alpha0, phi0)
    J4 = J_n_m(1, k1, alpha1, phi1)

    b = np.array([delta_C(1, rs), delta_C(0, rs)])
    # Mmat = np.array([[c0, c1],
    #                 [J0, J1]])
    Mmat = np.array([[J3, J4], [J0, J1]])
    # Solve for B0, B1
    B0, B1 = np.linalg.solve(Mmat, b)

    if get_constraints:
        return B0, B1
    else:
        delta_chi = B0 * np.exp(-alpha0 * r) * np.cos(k0 * r + phi0) + B1 * np.exp(
            -alpha1 * r
        ) * np.cos(k1 * r + phi1)
        return delta_chi
