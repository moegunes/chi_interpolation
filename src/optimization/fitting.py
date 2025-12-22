import numpy as np
from scipy.optimize import curve_fit


def guess_AB(r, kF, exact, model, initial_guess, kFr0=0, kFr1=8):
    # kF = gas.kF
    fit_idx0 = np.argmin(np.abs(kF * r - kFr0))
    fit_idx1 = np.argmin(np.abs(kF * r - kFr1))

    rr = r[fit_idx0:fit_idx1]

    def model_wrapper(r, B):
        # params = [gamma0, f0, phi0, gamma1, f1, phi1]
        return model(r, B, kF)

    p_opt, p_cov = curve_fit(
        model_wrapper,
        rr,
        exact[fit_idx0:fit_idx1],
        maxfev=30000,
    )
    return p_opt, p_cov
