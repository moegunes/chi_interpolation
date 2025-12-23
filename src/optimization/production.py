import numpy as np

from analysis.physics import get_B


def chi_interp(r, B, X, rs, gamma):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    S = X + B * np.sin(2 * kF * r) / r / r ** (gamma - 1)
    J = S - B * 2 * kF * np.cos(2 * kF * r) / r ** (gamma - 1)
    chiR = J * r / (2 * kF * r) ** 4 * r ** (gamma - 1)
    return chiR


def chi_interp2(r, B, X, rs):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    S = X + B * np.sin(2 * kF * r) / r**2
    J = S - B * 2 * kF * np.cos(2 * kF * r) / r
    chiR = J * r**2 / (2 * kF * r) ** 4
    return chiR


def get_chi_interp(r, params_dict, rs):
    model = params_dict["model"]
    gamma = params_dict["gamma"]
    params = params_dict[rs]
    B = get_B(rs)
    X = model(r, rs=rs, params=params, gamma=gamma)

    return chi_interp(r, B, X, rs, gamma)


def get_chi_interp2(r, params_dict, rs):
    model = params_dict["model"]
    gamma = params_dict["gamma"]
    params = params_dict[rs]
    B = get_B(rs)
    X = model(r, rs=rs, params=params, gamma=gamma)

    return chi_interp2(r, B, X, rs)
