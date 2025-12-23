import numpy as np
from scipy.optimize import curve_fit

from analysis.physics import get_B, get_chi
from optimization.models import X_r2_two_mode


def fit_params(rslist, q, r, model=X_r2_two_mode, inverse=False, gamma=1):
    from tqdm import tqdm

    parameters = {}
    parameters_cov = {}
    parameters["model"] = model
    parameters["gamma"] = gamma
    if inverse:
        rslist = rslist[::-1]
    for idx_rs in tqdm(range(len(rslist)), desc="Fitting", ncols=80):
        rs = rslist[idx_rs]
        kF = (9 * np.pi / 4) ** (1 / 3) / rs
        chiR = get_chi(q, rs)
        B = get_B(rs)

        X_exact = (
            chiR * (2 * kF * r) ** 4 / r
            + B * 2 * kF * np.cos(2 * kF * r)
            - B * np.sin(2 * kF * r) / r
        ) / r ** (gamma - 1)

        if idx_rs == 0:
            if inverse:
                initial_guess = [
                    0.26556495,
                    0.02491398,
                    5.02993162,
                    0.11291475,
                    0.05251752,
                    -0.29565569,
                ]
            else:
                initial_guess = [
                    1,
                    2 * kF / (2 * np.pi),
                    0.3,
                    2 * kF / (2 * np.pi),
                ]
                # initial_guess = [
                #    1,
                #    2 * kF / (2 * np.pi),
                #    np.pi / 2,
                #    0.3,
                #    2 * kF / (2 * np.pi),
                #    np.pi / 2 - 1e-4,
                # ]
        else:
            initial_guess = parameters[rslist[idx_rs - 1]]

        p_opt, p_cov = guess_X(
            r, rs, X_exact, model, initial_guess, gamma, kFr0=0, kFr1=8
        )

        parameters[rs] = p_opt
        parameters_cov[rs] = p_cov
    return parameters, parameters_cov


def guess_X(r, rs, X_exact, model, initial_guess, gamma, kFr0=0, kFr1=8):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    fit_idx0 = np.argmin(np.abs(kF * r - kFr0))
    fit_idx1 = np.argmin(np.abs(kF * r - kFr1))

    def model_wrapper(r, alpha0, f0, alpha1, f1):
        params = [alpha0, f0, alpha1, f1]
        return model(r, rs=rs, params=params, gamma=gamma)

    p_opt, p_cov = curve_fit(
        model_wrapper,
        r[fit_idx0:fit_idx1],
        X_exact[fit_idx0:fit_idx1],
        p0=initial_guess,
        maxfev=30000,
    )

    return p_opt, p_cov


def guess_AB(r, kF, exact, model, initial_guess, kFr0=24, kFr1=39):
    # kF = gas.kF
    fit_idx0 = np.argmin(np.abs(kF * r - kFr0))
    fit_idx1 = np.argmin(np.abs(kF * r - kFr1))

    rr = r[fit_idx0:fit_idx1]

    def model_wrapper(r, B):
        return model(r, B, kF)

    p_opt, p_cov = curve_fit(
        model_wrapper,
        rr,
        exact[fit_idx0:fit_idx1],
        maxfev=30000,
    )
    return p_opt, p_cov
