import numpy as np

from utils.utils_chi import get_chi0, get_gas_params, get_pi


def pi_interp(q, delta_pi, rs):
    from input import r

    kF, n0, NF = get_gas_params(rs)
    piR = get_pi(q, rs)
    chi0R = get_chi0(r, rs)

    piR = chi0R - 6 * np.pi * n0 * NF * delta_pi
    return piR


def get_pi_interp(r, q, params_dict, rs):
    model = params_dict["model"]
    params = params_dict[rs]
    delta_pi = model(r, rs=rs, params=params)
    return pi_interp(q, delta_pi, rs)
