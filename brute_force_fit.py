"""Quick brute-force test: what's the best fit at each low rs with kfR_max=15?"""

import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, "src")

from optimization.fitting import BOUNDS_LOWER, BOUNDS_UPPER
from optimization.models import delta_pi
from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import get_chi0, get_chi02, get_gas_params, get_pi, get_piq

N_q = 4096 * 16
q_max = 240.0
dq = q_max / N_q
q = np.arange(1, N_q + 1) * dq
r_grid = np.arange(1, N_q + 1) * np.pi / ((N_q + 1) * dq)

test_rs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0]

for rs in test_rs:
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    piR = get_pi(q, rs)
    chi0R = get_chi02(q, rs)
    delta_pi_exact = -(chi0R - piR) / factor
    pi_exact_q = get_piq(q, rs)

    def evaluate_params(params):
        dpi_r = delta_pi(r_grid, rs=rs, params=params)
        pi_r = get_chi0(r_grid, rs) + factor * dpi_r
        _, pi_q = chi_q_from_chi_r_fast(r_grid, pi_r)
        q_mask = q < 10.0 * kF
        return np.max(np.abs(pi_q[q_mask] - pi_exact_q[q_mask])) / NF

    best_err = {6: 1e30, 10: 1e30, 15: 1e30, 20: 1e30}
    best_params_all = {}

    for kfr_max_test in [6.0, 10.0, 15.0, 20.0]:
        i0 = np.argmin(np.abs(kF * r_grid - 0))
        i1 = np.argmin(np.abs(kF * r_grid - kfr_max_test))
        rf = r_grid[i0:i1]
        yf = delta_pi_exact[i0:i1]
        step = max(1, len(rf) // 4000)
        rf, yf = rf[::step], yf[::step]

        for trial in range(100):
            p0 = np.array(
                [
                    np.exp(np.random.uniform(np.log(0.001), np.log(5.0))),
                    np.random.uniform(0.02, 0.5),
                    np.random.uniform(-np.pi, np.pi),
                    np.exp(np.random.uniform(np.log(0.001), np.log(5.0))),
                    np.random.uniform(0.02, 0.5),
                    np.random.uniform(-np.pi, np.pi),
                ]
            )
            try:

                def mw(r, a0, f0, ph0, a1, f1, ph1):
                    return delta_pi(r, rs=rs, params=[a0, f0, ph0, a1, f1, ph1])

                p0c = np.clip(p0, BOUNDS_LOWER + 1e-6, BOUNDS_UPPER - 1e-6)
                popt, _ = curve_fit(
                    mw,
                    rf,
                    yf,
                    p0=p0c,
                    bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
                    method="trf",
                    maxfev=30000,
                )
                err = evaluate_params(popt)
                k = int(kfr_max_test)
                if err < best_err[k]:
                    best_err[k] = err
                    best_params_all[k] = popt.copy()
            except Exception:
                pass

    print(
        f"rs={rs:.2f}:  "
        + "  ".join(f"kfR={k}: {best_err[k]:.6f}" for k in [6, 10, 15, 20])
    )
