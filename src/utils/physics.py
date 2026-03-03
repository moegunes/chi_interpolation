import math

import numpy as np

from utils.io import load_dict
from utils.utils_chi import G_Moroni, corradini_pz


def get_gas_params(rs):
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (1 * np.pi**2)
    return kF, n0, NF


def I_n_m(n, k, gamma, phi):
    """Integral I_n^m = \int_0^\infty r^(2n+1) exp(-\gamma r) cos(kr + \phi) dr. Equivalent to function J_n^m in ..."""
    # Compute A_n, B_n
    A, B = compute_A_B(n, k, gamma)
    Lambda = math.factorial(2 * n + 1) / (gamma**2 + k**2) ** (2 * n + 2)
    return Lambda * (A * np.cos(phi) - B * np.sin(phi))


def J_n_m(n, k, gamma, phi):
    """Integral I_n^m = \int_0^\infty r^(2n+2) exp(-\gamma r) cos(kr + \phi) dr. Equivalent to function J_n^m in ..."""
    return math.factorial(2 * n + 2) * np.real(
        np.exp(1j * phi) / (gamma - 1j * k) ** (2 * n + 3)
    )


def J_n_m_kFr(n, k, gamma, phi, kF):
    """Integral I_n^m = \int_0^\infty r^(2n+2) exp(-\gamma r) cos(kr + \phi) dr. Equivalent to function J_n^m in ..."""
    return math.factorial(2 * n + 2) * np.real(
        np.exp(1j * phi) / (gamma - 1j * k) ** (2 * n + 3) / kF ** (2 * n + 3)
    )


def compute_A_B(n, k, gamma):
    """Compute A_n(k, gamma) and B_n(k, gamma) explicitly."""
    A = 0.0
    B = 0.0
    n = int(n)
    # A_n sum: even powers
    for j in range(n + 1 + 1):  # j = 0 ... n+1
        coeff = (-1) ** j * math.comb(2 * n + 2, 2 * j)
        A += coeff * (gamma ** (2 * n + 2 - 2 * j)) * (k ** (2 * j))

    # B_n sum: odd powers
    for j in range(n + 1):  # j = 0 ... n
        coeff = (-1) ** j * math.comb(2 * n + 2, 2 * j + 1)
        B += coeff * (gamma ** (2 * n + 1 - 2 * j)) * (k ** (2 * j + 1))

    return A, B


def canon_cos_phase(phi):
    phi = np.mod(phi, 2 * np.pi)
    return np.minimum(phi, 2 * np.pi - phi)


def chi_moment(n, rs):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    f0 = corradini_pz(rs, 0)
    if n == 0:
        return 0
    if n == 1:
        return 3 / (8 * np.pi**2)
    if n == 2:
        return 15 / (8 * np.pi * kF) + 15 * f0 / (8 * np.pi**3)
    if n == 3:

        def f1_corradini(rs, dq):
            q = np.array([-dq, 0.0, dq])
            f = corradini_pz(rs, q)
            fpp = (f[2] - 2 * f[1] + f[0]) / dq**2
            return 0.5 * fpp

        f1 = f1_corradini(rs, 1e-3)
        der = (
            -45
            / (4 * kF**2 * np.pi**3)
            * (
                f0**2 * kF**2
                - 4 * f1 * kF**2 * np.pi
                + 2 * f0 * kF * np.pi**2
                + np.pi**4
            )
        )
        return -1 / (4 * np.pi) * 7 * der


def pi_moment(n, rs):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs

    # f0 = f_xc(0)
    f0 = corradini_pz(rs, 0.0)

    def f2_corradini(rs, dq):
        q = np.array([-dq, 0.0, dq])
        fvals = corradini_pz(rs, q)
        fpp = (fvals[2] - 2 * fvals[1] + fvals[0]) / dq**2
        return 0.5 * fpp

    def f4_corradini(rs, dq):
        q = np.array([-2 * dq, -dq, 0.0, dq, 2 * dq])
        fvals = corradini_pz(rs, q)

        f4_raw = (
            fvals[0] - 4 * fvals[1] + 6 * fvals[2] - 4 * fvals[3] + fvals[4]
        ) / dq**4

        return f4_raw / 24.0

    def f6_corradini(rs, dq):
        q = np.array([-3 * dq, -2 * dq, -dq, 0.0, dq, 2 * dq, 3 * dq])
        fvals = corradini_pz(rs, q)

        f6_raw = (
            -fvals[0]
            + 6 * fvals[1]
            - 15 * fvals[2]
            + 20 * fvals[3]
            - 15 * fvals[4]
            + 6 * fvals[5]
            - fvals[6]
        ) / dq**6

        return f6_raw / 720.0  # divide by 6!

    f2 = f2_corradini(rs, 1e-3)
    f4 = f4_corradini(rs, 1e-3)
    f6 = f6_corradini(rs, 1e-3)

    denom = f0 * kF + np.pi**2

    if n == 0:
        return -kF / (4 * np.pi * denom)

    if n == 1:
        return -(12 * f2 * kF**3 + np.pi**2) / (8 * kF * np.pi * denom**2)

    if n == 2:
        numerator = (
            -720 * (f2**2 - f0 * f4) * kF**6
            + 8 * kF * (f0 - 15 * f2 * kF**2 + 90 * f4 * kF**4) * np.pi**2
            + 3 * np.pi**4
        )

        return numerator / (24 * kF**3 * np.pi * denom**3)

    if n == 3:
        numerator = (
            60480 * (f2**3 - 2 * f0 * f2 * f4 + f0**2 * f6) * kF**9
            + 3
            * kF**2
            * (
                29 * f0**2
                + 5040 * f2 * kF**4 * (f2 - 8 * f4 * kF**2)
                - 224 * f0 * (2 * f2 * kF**2 + 15 * kF**4 * (f4 - 12 * f6 * kF**2))
            )
            * np.pi**2
            + 2
            * kF
            * (31 * f0 - 42 * kF**2 * (f2 + 120 * kF**2 * (f4 - 6 * f6 * kF**2)))
            * np.pi**4
            + 10 * np.pi**6
        )

        return -numerator / (48 * kF**5 * np.pi * denom**4)

    raise ValueError("n must be 0,1,2,3")


def chi0_moment(n, rs):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    if n == 0:
        return -kF / (4 * np.pi**3)
    if n == 1:
        return -1 / (8 * np.pi**3 * kF)
    if n == 2:
        return 1 / (8 * np.pi**3 * kF**3)
    if n == 3:
        return -1 / (4 * np.pi) * 5 / (6 * kF**5 * np.pi**2)


def K(n, rs):
    B = get_B(rs)
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    return 16 * kF**4 * (chi_moment(n, rs) - B / factor * chi0_moment(n, rs))


def delta_C(n, rs):
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    return (pi_moment(n, rs) - chi0_moment(n, rs)) / factor


def get_B(rs):
    kF, n0, NF = get_gas_params(rs)
    delta = 32 * (4 * np.pi) ** (1 / 3) * rs**2 / (81 * np.pi)
    kappa = np.sqrt(4 / np.pi * (9 * np.pi / 4) ** (1 / 3) / rs)
    Gplus = G_Moroni(rs, 2 * kF)
    beta = (
        delta
        / (1 + 2 * kappa**2 / kF**2 / 16 * (1 - Gplus)) ** 2
        * n0
        * 8
        * kF**3
        * 0.221702924555749
    )  # (4*pi/9)**(1/3)/5
    return -beta


def get_B2(rs):
    Bdict = load_dict("B_dict")
    return Bdict[rs]
