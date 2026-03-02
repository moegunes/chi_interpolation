"""Cross-validate chi_mhg against the original chi_interpolation codebase."""

import os
import sys

os.chdir("/Users/muhammedxgunes/Desktop/research/chi_interpolation")
sys.path.insert(0, "/Users/muhammedxgunes/Desktop/research/chi_interpolation/src")

import numpy as np
from chi_mhg import chi0_heg, delta_chi_mhg
from chi_mhg.core import _gas_params

from optimization.models import delta_chi as delta_chi_orig
from utils.io import load_dict
from utils.utils_chi import get_chi0 as chi0_orig

params_dict = load_dict("parameters")
rsl = sorted(
    [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
)

r = np.linspace(0.1, 50, 3000)

print(f"{'rs':>6s} | {'max|Δchi|/NF':>14s} | {'max|chi0 diff|':>14s} | {'status'}")
print("-" * 60)

for rs in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    kF, n0, NF = _gas_params(rs)
    factor = -6.0 * np.pi * n0 * NF

    # Original chi0
    c0_orig = chi0_orig(r, rs)
    c0_new = chi0_heg(r, rs)
    chi0_diff = np.max(np.abs(c0_orig - c0_new))

    # Original delta_chi with fitted params
    if rs in params_dict:
        params_orig = params_dict[rs]
        dchi_orig = delta_chi_orig(r, rs=rs, params=params_orig)

        # New delta_chi with interpolated params
        dchi_new = delta_chi_mhg(r, rs)

        mask = kF * r < 15
        diff = np.abs(factor * (dchi_new - dchi_orig))
        max_err = np.max(diff[mask]) / NF
        status = "OK" if max_err < 0.02 else "CHECK"
    else:
        max_err = float("nan")
        status = "no ref"

    print(f"{rs:6.1f} | {max_err:14.6f} | {chi0_diff:14.2e} | {status}")
