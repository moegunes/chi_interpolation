import numpy as np


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
