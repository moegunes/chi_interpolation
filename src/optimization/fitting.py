import numpy as np
from scipy.optimize import curve_fit

from optimization.models import delta_chi
from utils.io import load_dict
from utils.utils_chi import get_chi, get_chi02, get_gas_params


def fit_params(rslist, q, r, model=delta_chi, inverse=False):
    from tqdm import tqdm

    parameters = {}
    parameters_cov = {}
    parameters["model"] = model
    if inverse:
        rslist = rslist[::-1]
    for idx_rs in tqdm(range(len(rslist)), desc="Fitting", ncols=80):
        rs = rslist[idx_rs]
        kF, n0, NF = get_gas_params(rs)
        factor = -6 * np.pi * n0 * NF
        chiR = get_chi(q, rs)
        chi0R = get_chi02(q, rs)

        delta_chi_exact = -(chi0R - chiR) / factor

        if idx_rs == 0:
            if inverse:
                params_temp = load_dict("parameters")
                print(
                    'Inverse fit: using parameters from "parameters" as initial guess. rs = ',
                    10,
                )
                initial_guess = params_temp[10]
            else:
                initial_guess = [
                    0.5,
                    2 * kF / 2.0 / np.pi,
                    np.pi / 2 - 0.1,
                    1,
                    -1 * kF / 2.0 / np.pi,
                    0.01,
                ]
        else:
            initial_guess = parameters[rslist[idx_rs - 1]]

        p_opt, p_cov = guess_X(
            r,
            rs,
            delta_chi_exact,
            model,
            initial_guess,
            kFr0=0,
            kFr1=8,
        )

        parameters[rs] = p_opt
        parameters_cov[rs] = p_cov
    return parameters, parameters_cov


def guess_X(r, rs, X_exact, model, initial_guess, kFr0=0, kFr1=4):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    fit_idx0 = np.argmin(np.abs(kF * r - kFr0))
    fit_idx1 = np.argmin(np.abs(kF * r - kFr1))

    def model_wrapper(r, alpha0, f0, phi0, alpha1, f1, phi1):
        params = [alpha0, f0, phi0, alpha1, f1, phi1]
        return model(r, rs=rs, params=params)

    p_opt, p_cov = curve_fit(
        model_wrapper,
        r[fit_idx0:fit_idx1],
        X_exact[fit_idx0:fit_idx1],
        p0=initial_guess,
        maxfev=30000,
    )

    return p_opt, p_cov
