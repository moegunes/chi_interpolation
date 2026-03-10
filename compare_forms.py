"""Compare parametric forms: error vs EXACT pi (Corradini-PZ).

For each form:
1. Fit 6 params with single curve_fit (no restarts) — production pipeline
2. For each test rs, interpolate params via the fitted form
3. Build pi(r) from the two-damped-cosine model
4. FT to q-space via DST-I
5. Compare against exact pi(q) from get_piq
"""

import os
import sys

os.chdir("/Users/muhammedxgunes/Desktop/research/chi_interpolation")
sys.path.insert(0, "src")

import numpy as np

from optimization.models import delta_pi
from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import get_chi0, get_gas_params, get_piq
from visualization.pp import (
    PARAMETRIC_FORMS,
    fit_all_parameters,
    get_interpolated_params,
)

# DST-I grid
N_q = 4096 * 16
q_max = 1240.0
dq = q_max / N_q
q = np.arange(1, N_q + 1) * dq
m = np.arange(1, N_q + 1)
r_grid = m * np.pi / ((N_q + 1) * dq)

# Load reference parameters
from utils.io import load_dict

params_dict = load_dict("parameters")

# Test rs points — dense grid
test_rs = np.round(np.arange(0.5, 10.01, 0.25), 2).tolist()

# Forms to compare
form_names = ["mPZ[2/3]√", "mPZ[2/3]", "PZ[2/3]√", "PZ[2/3]", "PZ[2/2]√", "Pade[2/2]"]

results = {}

for form_name in form_names:
    print(f"\n{'=' * 60}")
    print(f"Form: {form_name}")
    print(f"{'=' * 60}")

    # Step 1: Fit parametric forms to the 6 parameters
    fits = fit_all_parameters(params_dict, form_name=form_name)

    # Check for failures
    failed = []
    for pname in ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]:
        entry = fits[pname]
        if entry["popt"] is None:
            failed.append(f"{pname}: {entry.get('error', '?')}")
            print(f"  FAILED {pname}: {entry.get('error', '?')}")
        else:
            print(f"  {pname}: maxpct={entry['maxpct']:.2f}%")

    if failed:
        print(
            f"  >>> SKIPPING (failed params: {', '.join(p.split(':')[0] for p in failed)})"
        )
        results[form_name] = None
        continue

    # Step 2-5: For each rs, interpolate -> build pi(r) -> FT -> compare vs exact
    all_max_q = []
    all_made_q = []

    for rs in test_rs[:]:
        kF, n0, NF = get_gas_params(rs)
        factor = -6.0 * np.pi * n0 * NF

        # Interpolated params from fitted form
        interp_params = get_interpolated_params(rs, fits)

        # Build pi(r) on DST grid
        dpi_r = delta_pi(r_grid, rs=rs, params=interp_params)
        pi_interp_r = get_chi0(r_grid, rs) + factor * dpi_r

        # FT to q
        _, pi_interp_q = chi_q_from_chi_r_fast(r_grid, pi_interp_r)

        # Exact pi(q)
        pi_exact_q = get_piq(q, rs)

        # Metrics
        q_mask = q < 10.0 * kF
        diff_q = np.abs(pi_interp_q - pi_exact_q)
        max_dpiq_NF = np.max(diff_q[q_mask]) / NF

        ref_abs_q = np.abs(pi_exact_q[q_mask])
        valid_q = ref_abs_q > 1e-6 * NF
        if np.any(valid_q):
            made_q = np.sum(diff_q[q_mask][valid_q]) / np.sum(ref_abs_q[valid_q]) * 100
        else:
            made_q = float("nan")

        all_max_q.append(max_dpiq_NF)
        all_made_q.append(made_q)

    ncoeffs = PARAMETRIC_FORMS[form_name][1] * 6
    worst_max = max(all_max_q)
    mean_made = np.mean(all_made_q)
    worst_made = max(all_made_q)
    results[form_name] = {
        "worst_max": worst_max,
        "mean_made": mean_made,
        "worst_made": worst_made,
        "ncoeffs": ncoeffs,
        "all_max_q": all_max_q,
        "all_made_q": all_made_q,
    }
    print(f"  worst max|dpiq|/NF = {worst_max:.6f}")
    print(f"  worst MADE = {worst_made:.4f}%,  mean MADE = {mean_made:.4f}%")
    print(f"  ncoeffs = {ncoeffs}")

# Final ranking
print("\n\n" + "=" * 80)
print("RANKING by worst max|δχ(q)|/NF  (vs EXACT pi)")
print("=" * 80)
print(
    f"{'#':>2s}  {'Form':<14s}  {'worst max/NF':>12s}  {'mean MADE':>10s}  {'worst MADE':>11s}  {'nc':>3s}"
)
print("-" * 65)

ranked = [(k, v) for k, v in results.items() if v is not None]
ranked.sort(key=lambda x: x[1]["worst_max"])

for i, (fname, r) in enumerate(ranked, 1):
    print(
        f"{i:2d}  {fname:<14s}  {r['worst_max']:12.6f}  {r['mean_made']:9.4f}%  {r['worst_made']:10.4f}%  {r['ncoeffs']:3d}"
    )

# Print failed forms
for fname, r in results.items():
    if r is None:
        print(f" -  {fname:<14s}  {'FAILED':>12s}")
