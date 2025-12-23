import numpy as np
from scipy.optimize import curve_fit, least_squares

from analysis.physics import get_B, get_chi
from optimization.models import X_r2_two_mode, moment_residuals


def fit_params(
    rslist, q, r, model=X_r2_two_mode, inverse=False, gamma=1, fit_residue=False
):
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
                    np.pi / 2,
                    0.3,
                    2 * kF / (2 * np.pi),
                    np.pi / 2 - 1e-4,
                ]
        else:
            initial_guess = parameters[rslist[idx_rs - 1]]

        if fit_residue == "moment":
            p_opt, p_cov = guess_X_moments(rs, initial_guess, n_residuals=(2,))

        elif fit_residue == "hybrid":
            res = least_squares(
                hybrid_residuals,
                x0=initial_guess,
                args=(r, rs, X_exact, gamma, (2,), 1),
                max_nfev=30000,
            )
            p_opt = res.x
            p_cov = None

        else:  # pure r-space
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

    def model_wrapper(r, alpha0, f0, phi0, alpha1, f1, phi1):
        params = [alpha0, f0, phi0, alpha1, f1, phi1]
        return model(r, rs=rs, params=params, gamma=gamma)

    p_opt, p_cov = curve_fit(
        model_wrapper,
        r[fit_idx0:fit_idx1],
        X_exact[fit_idx0:fit_idx1],
        p0=initial_guess,
        maxfev=30000,
    )

    return p_opt, p_cov


def guess_X_moments(rs, initial_guess, n_residuals=(2,)):
    res = least_squares(
        moment_residuals,
        x0=initial_guess,
        args=(rs, n_residuals),
        max_nfev=30000,
    )
    return res.x, res


def hybrid_residuals(params, r, rs, X_exact, gamma, n_residuals=(2, 3), w_moment=1.0):
    # r-space residual
    X_model = X_r2_two_mode(r, rs, params, gamma)
    res_r = X_model - X_exact

    # moment residual
    res_m = moment_residuals(params, rs, n_residuals)

    return np.concatenate([res_r, w_moment * res_m])


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
