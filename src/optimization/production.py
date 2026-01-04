import numpy as np

from utils.utils_chi import get_chi, get_chi02, get_gas_params


def chi_interp(q, delta_chi, rs):
    kF, n0, NF = get_gas_params(rs)
    chiR = get_chi(q, rs)
    chi0R = get_chi02(q, rs)

    chiR = chi0R - 6 * np.pi * n0 * NF * delta_chi
    return chiR


def get_chi_interp(r, q, params_dict, rs):
    model = params_dict["model"]
    params = params_dict[rs]
    delta_chi = model(r, rs=rs, params=params)
    return chi_interp(q, delta_chi, rs)
