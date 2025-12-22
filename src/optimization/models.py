import numpy as np

from analysis.physics import I_n_m, K


def X_r2_two_mode(r, B, rs, factor, params, get_constraints=False):
    """
    Two-mode model with r^2 prefactor:
        X(r) = r^2 [ B0 e^{-\gamma_0 r} cos(k0 r + \phi_0)
                   + B1 e^{-\gamma_1 r} cos(k1 r + \phi_1) ]

    Parameters (fit):
        params = [\gamma_0, f0, \phi_0, \gamma_1, f1, \phi_1]

    B0, B1 are determined by:
        sum B_m cos(φ_m) = C2'
        sum B_m J_m      = 2 kF B
    """
    r = np.asarray(r, float)
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    gamma0, f0, phi0, gamma1, f1, phi1 = params
    phi0, phi1 = np.mod(phi0, 2 * np.pi), np.mod(phi1, 2 * np.pi)
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1

    # Second derivative constraint: X''(0) = C2
    C2 = (16.0 / 3.0) * kF**3 * (factor - B)
    C2p = 0.5 * C2  # sum B_m cosφ_m = C2/2
    n = 0
    # C2p=K(n,rs,B)
    # Coeffs
    c0 = np.cos(phi0)
    c1 = np.cos(phi1)
    J0 = I_n_m(n, k0, gamma0, phi0)
    J1 = I_n_m(n, k1, gamma1, phi1)
    # J3 = I_n_m(n, k0, gamma0, phi0)
    # J4 = I_n_m(n, k1, gamma1, phi1)

    # RHS of constraints
    S1 = K(n, rs, B)  # -2.0 * kF * B   # sum B_m J_m = 2 kF B

    b = np.array([C2p, S1])
    Mmat = np.array([[c0, c1], [J0, J1]])
    # Mmat = np.array([[J3, J4],
    #                 [J0, J1]])
    # Solve for B0, B1
    B0, B1 = np.linalg.solve(Mmat, b)

    if get_constraints:
        return B0, B1
    else:
        X = r**2 * (
            B0 * np.exp(-gamma0 * r) * np.cos(k0 * r + phi0)
            + B1 * np.exp(-gamma1 * r) * np.cos(k1 * r + phi1)
        )
        return X


def X_r2_two_mode_2(r, B, rs, factor, params, get_constraints=False):
    """
    Two-mode model with r^2 prefactor:
        X(r) = r^2 [ B0 e^{-\gamma_0 r} cos(k0 r + \phi_0)
                   + B1 e^{-\gamma_1 r} cos(k1 r + \phi_1) ]

    Parameters (fit):
        params = [\gamma_0, f0, \phi_0, \gamma_1, f1, \phi_1]

    B0, B1 are determined by:
        sum B_m cos(φ_m) = C2'
        sum B_m J_m      = 2 kF B
    """
    r = np.asarray(r, float)
    gamma0, f0, phi0, gamma1, f1, phi1 = params
    phi0, phi1 = np.mod(phi0, 2 * np.pi), np.mod(phi1, 2 * np.pi)
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1

    # Second derivative constraint: X''(0) = C2
    # C2  = (16.0/3.0) * kF**3 * (factor - B)
    # C2p = 0.5 * C2   # sum B_m cosφ_m = C2/2
    n = 1
    # Coeffs
    J0 = I_n_m(0, k0, gamma0, phi0)
    J1 = I_n_m(0, k1, gamma1, phi1)
    J3 = I_n_m(n, k0, gamma0, phi0)
    J4 = I_n_m(n, k1, gamma1, phi1)

    b = np.array([K(n, rs, B), K(0, rs, B)])
    # Mmat = np.array([[c0, c1],
    #                 [J0, J1]])
    Mmat = np.array([[J3, J4], [J0, J1]])
    # Solve for B0, B1
    B0, B1 = np.linalg.solve(Mmat, b)

    if get_constraints:
        return B0, B1
    else:
        X = r**2 * (
            B0 * np.exp(-gamma0 * r) * np.cos(k0 * r + phi0)
            + B1 * np.exp(-gamma1 * r) * np.cos(k1 * r + phi1)
        )
        return X


def X_2(r, B, rs, factor, params, get_constraints=False):
    """
    Two-mode model with r prefactor:
        X(r) = r [ B0 e^{-\gamma_0 r} cos(k0 r + \phi_0)
                   + B1 e^{-\gamma_1 r} cos(k1 r + \phi_1) ]

    Parameters (fit):
        params = [\gamma_0, f0, \phi_0, \gamma_1, f1, \phi_1]
    """
    r = np.asarray(r, float)
    gamma0, f0, phi0, gamma1, f1, phi1 = params
    phi0, phi1 = np.mod(phi0, 2 * np.pi), np.mod(phi1, 2 * np.pi)
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1

    n = 1
    J0 = I_n_m(0, k0, gamma0, phi0)
    J1 = I_n_m(0, k1, gamma1, phi1)
    J3 = I_n_m(n, k0, gamma0, phi0)
    J4 = I_n_m(n, k1, gamma1, phi1)

    b = np.array([K(n, rs, B), K(0, rs, B)])
    # Mmat = np.array([[c0, c1],
    #                 [J0, J1]])
    Mmat = np.array([[J3, J4], [J0, J1]])
    # Solve for B0, B1
    B0, B1 = np.linalg.solve(Mmat, b)

    if get_constraints:
        return B0, B1
    else:
        X = r * (
            B0 * np.exp(-gamma0 * r) * np.cos(k0 * r + phi0)
            + B1 * np.exp(-gamma1 * r) * np.cos(k1 * r + phi1)
        )
        return X


def model2(r, A, rs, B, k1, k2):
    from analysis.physics import corradini_pz, get_gas_params

    kF, n0, NF = get_gas_params(rs)
    # Safe handling for r=0 if needed
    r = np.array(r)
    delta = 32 * (4 * np.pi) ** (1 / 3) * rs**2 / (81 * np.pi)
    kappa = np.sqrt(4 / np.pi * (9 * np.pi / 4) ** (1 / 3) / rs)
    Gplus = -corradini_pz(rs, 2 * kF) / (4 * np.pi / (2 * kF) ** 2)  # G_Moroni(rs,2*kF)

    beta = -delta / (1 + 2 * kappa**2 / kF**2 / 16 * (1 - Gplus)) ** 2 * n0 * 2 * kF**3

    tune = 0.05
    B_s = (B - beta) / (tune * r + 1) + beta  # ---> beta for high r
    # C_s=(C+tune*r)/(tune*r+1)
    A_s = (A - beta) / (tune * r + 1) + beta
    k1_s = (k1 - 2 * kF) / (tune * r + 1) + 2 * kF
    k2_s = (k2 - 2 * kF) / (tune * r + 1) + 2 * kF
    # A*(np.sin(k1*(r-r1)) - (r-r2)*np.cos(k2*(r-r2)))
    return A * np.sin(2 * kF * r) - B * 2 * kF * r * np.cos(2 * kF * r)


def model3(r, B, kF):
    # Safe handling for r=0 if needed
    return -B * 2 * kF * np.cos(2 * kF * r)


def model4(r, A, kF):
    return A * np.sin(2 * kF * r) / r
