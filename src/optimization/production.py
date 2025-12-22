import numpy as np

from analysis.physics import get_B


def chi_interp(r, B, X, rs):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    S = X + B * np.sin(2 * kF * r) / r
    J = S - B * 2 * kF * np.cos(2 * kF * r)
    chiR = J * r / (2 * kF * r) ** 4
    return chiR


def chi_interp2(r, B, X, rs):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    S = X + B * np.sin(2 * kF * r) / r**2
    J = S - B * 2 * kF * np.cos(2 * kF * r) / r
    chiR = J * r**2 / (2 * kF * r) ** 4
    return chiR


def get_chi_interp(r, params_dict, rs):
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (1 * np.pi**2)
    model = params_dict["model"]
    params = params_dict[rs]
    B = get_B(rs)
    factor = -6 * np.pi * n0 * NF
    X = model(r, B=B, rs=rs, factor=factor, params=params)

    return chi_interp(B, X, rs)


def get_chi_interp2(r, params_dict, rs):
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (1 * np.pi**2)
    model = params_dict["model"]
    params = params_dict[rs]
    B = get_B(rs)
    factor = -6 * np.pi * n0 * NF
    X = model(r, B=B, rs=rs, factor=factor, params=params)

    return chi_interp2(B, X, rs)
