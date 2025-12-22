import numpy as np

from analysis.physics import get_B
from optimization.models import X_r2_two_mode


def get_constraints(r, params, rs, model=X_r2_two_mode):
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (1 * np.pi**2)
    B = get_B(rs)
    factor = -6 * np.pi * n0 * NF
    return model(r, B, rs, factor, params, get_constraints=True)
