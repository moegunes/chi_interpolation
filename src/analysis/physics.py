import numpy as np

from utils.utils_chi import get_chi


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


def canon_cos_phase(phi):
    phi = np.mod(phi, 2 * np.pi)
    return np.minimum(phi, 2 * np.pi - phi)
