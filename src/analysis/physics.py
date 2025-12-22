import math

import numpy as np

from utils.io import load_dict
from utils.utils_chi import corradini_pz, get_chi


class ElectronGas:
    def __init__(self, rs):
        self.rs = rs
        self.kF, self.n0, self.NF = get_gas_params(rs)
        self.factor = 6 * np.pi * self.n0 * self.NF
        self.qmax = 20000
        self.dq = 0.01
        self.q, self.r = self._get_qr()
        self.chiR = get_chi(self.q, rs)
        self.chi0R = self._get_chi0()

    def _get_chi0(self):
        r = self.r
        kF = self.kF
        factor = self.factor
        chi0R = (
            -factor
            * (np.sin(2 * kF * (r)) - 2 * kF * r * np.cos(2 * kF * (r)))
            / (2 * kF * (r + 1e-15)) ** 4
        )
        return chi0R

    def _get_qr(self):
        from utils.utils_chi import chi00q, chi_r_from_chi_q_fast

        rs = self.rs

        qmax = self.qmax
        dq = self.dq
        q = np.arange(
            0.0, qmax + dq / 2, dq
        )  # starts at 0; code will drop q=0 internally
        chi0 = chi00q(q, rs)

        # Get \chi(r) on the fast-transform’s dual grid:
        r = chi_r_from_chi_q_fast(q, chi0)[0]
        return q[1:], r


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
    fxc0 = corradini_pz(rs, 0)
    if n == 0:
        return 0
    if n == 1:
        return 3 / (8 * np.pi**2)
    if n == 2:
        return 15 / (16 * np.pi * kF) + 15 * fxc0 / (16 * np.pi**3)


def chi0_moment(n, rs):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    if n == 0:
        return -kF / (4 * np.pi**3)
    if n == 1:
        return -1 / (8 * np.pi**3 * kF)
    if n == 2:
        return 1 / (8 * np.pi**3 * kF**3)


def K(n, rs, B):
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (1 * np.pi**2)
    factor = -6 * np.pi * n0 * NF
    return 16 * kF**4 * (chi_moment(n, rs) - B / factor * chi0_moment(n, rs))


def K2(n, rs):
    B = get_B(rs)
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (1 * np.pi**2)
    factor = -6 * np.pi * n0 * NF
    return 16 * kF**4 * (chi_moment(n, rs) - B / factor * chi0_moment(n, rs))


def get_B(rs):
    Bdict = load_dict("B_dict")
    return Bdict[rs]
