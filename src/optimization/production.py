import numpy as np


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
