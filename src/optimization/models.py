import numpy as np

from utils.physics import I_n_m, J_n_m, K, delta_C


def delta_pi(r, rs, params, get_constraints=False):
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


def moment_residuals(params, rs, n_residuals=(2, 3)):
    """
    Residuals in moment space.
    n_residuals: moments NOT enforced exactly
    """
    alpha0, f0, phi0, alpha1, f1, phi1 = params
    phi0 = np.mod(phi0, 2 * np.pi)
    phi1 = np.mod(phi1, 2 * np.pi)
    k0 = 2 * np.pi * f0
    k1 = 2 * np.pi * f1

    # --- enforce two moments exactly (n=0,1) ---
    J0 = I_n_m(0, k0, alpha0, phi0)
    J1 = I_n_m(0, k1, alpha1, phi1)
    J3 = I_n_m(1, k0, alpha0, phi0)
    J4 = I_n_m(1, k1, alpha1, phi1)

    Mmat = np.array([[J3, J4], [J0, J1]])
    b = np.array([K(1, rs), K(0, rs)])

    B0, B1 = np.linalg.solve(Mmat, b)

    # --- residuals for higher moments ---
    res = []
    for n in n_residuals:
        In0 = I_n_m(n, k0, alpha0, phi0)
        In1 = I_n_m(n, k1, alpha1, phi1)
        res.append(B0 * In0 + B1 * In1 - K(n, rs))

    return np.array(res)


def model3(r, B, kF):
    # Safe handling for r=0 if needed
    return -B * 2 * kF * np.cos(2 * kF * r)


def model4(r, A, kF):
    return A * np.sin(2 * kF * r) / r
