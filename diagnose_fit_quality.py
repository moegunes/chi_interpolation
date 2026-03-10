"""Diagnose fitting quality per rs: compare Pi(q) from fitted params vs exact Pi(q).

This isolates the fitting error (parameters.pkl quality) from the form interpolation error.
"""

import pickle
import sys

import numpy as np

sys.path.insert(0, "src")

from optimization.models import delta_pi
from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import get_chi0, get_gas_params, get_piq

# DST grid (same as compare_forms.py)
N_q = 4096 * 16
q_max = 240.0
dq = q_max / N_q
q = np.arange(1, N_q + 1) * dq
m = np.arange(1, N_q + 1)
r_grid = m * np.pi / ((N_q + 1) * dq)

with open("parameters.pkl", "rb") as f:
    params_dict = pickle.load(f)

rsl = sorted([k for k in params_dict if isinstance(k, (int, float)) and k != "model"])

print(
    f"{'rs':>5s}  {'max|dPi(q)|/NF':>15s}  {'MADE%':>8s}  "
    f"{'alpha0':>8s}  {'f0':>8s}  {'phi0':>8s}  {'alpha1':>8s}  {'f1':>8s}  {'phi1':>8s}"
)
print("-" * 100)

bad_rs = []

for rs in rsl:
    kF, n0, NF = get_gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF

    params = params_dict[rs]
    dpi_r = delta_pi(r_grid, rs=rs, params=params)
    pi_r = get_chi0(r_grid, rs) + factor * dpi_r
    _, pi_q = chi_q_from_chi_r_fast(r_grid, pi_r)

    pi_exact = get_piq(q, rs)

    q_mask = q < 10.0 * kF
    diff = np.abs(pi_q[q_mask] - pi_exact[q_mask])
    max_err = np.max(diff) / NF

    ref = np.abs(pi_exact[q_mask])
    valid = ref > 1e-6 * NF
    made = (
        np.sum(diff[valid]) / np.sum(ref[valid]) * 100
        if np.any(valid)
        else float("nan")
    )

    p = params
    marker = " ***" if max_err > 0.02 else ""
    print(
        f"{rs:5.2f}  {max_err:15.6f}  {made:8.4f}  "
        f"{p[0]:8.4f}  {p[1]:8.4f}  {p[2]:8.4f}  {p[3]:8.4f}  {p[4]:8.4f}  {p[5]:8.4f}{marker}"
    )
    if max_err > 0.02:
        bad_rs.append((rs, max_err))

print()
if bad_rs:
    print(f"BAD rs points (max|dPi(q)|/NF > 0.02): {len(bad_rs)}")
    for rs, err in bad_rs:
        print(f"  rs={rs:.2f}  max_err/NF={err:.6f}")
else:
    print("All rs points have max|dPi(q)|/NF < 0.02")
