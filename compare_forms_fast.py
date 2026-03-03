"""Compare physically-consistent form maps for chi interpolation error.
Ultra-fast: compares delta_chi(r) directly — no FT needed at all."""

import sys
import time

import numpy as np

sys.path.insert(0, "src")
from numpy import pi

from utils.io import load_dict
from utils.utils_chi import get_gas_params
from visualization.pp import (
    PARAMETRIC_FORMS,
    fit_all_parameters,
    get_interpolated_params,
)

params_dict = load_dict("parameters")
model = params_dict["model"]
rsl = sorted(
    [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
)
rs_test = [rs for rs in rsl if rs >= 0.5]

# Use a modest r-grid for evaluation (no need for fine q-grid)
r_eval = np.linspace(0.01, 50, 5000)

# Precompute original delta_chi for all rs
print("Precomputing original delta_chi for all rs...")
orig_dchi = {}
for rs in rs_test:
    orig_dchi[rs] = model(r_eval, rs=rs, params=params_dict[rs])
print(f"Done. {len(rs_test)} rs points.\n")


def eval_form_map(fm, params_dict, rs_test, r_eval, orig_dchi):
    """Evaluate a form map by comparing delta_chi directly."""
    fits = fit_all_parameters(params_dict, form_name=fm)

    # Check for failed fits
    for pname in ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]:
        if fits[pname]["popt"] is None:
            return None, None, None, pname, 999, fits

    results = []
    for rs in rs_test[6:]:
        kF, n0, NF = get_gas_params(rs)
        factor = -6 * pi * n0 * NF

        params_interp = get_interpolated_params(rs, fits)
        dchi_interp = model(r_eval, rs=rs, params=params_interp)
        dchi_orig = orig_dchi[rs]

        # Normalized error: |factor * (dchi_interp - dchi_orig)| / NF
        diff = np.abs((dchi_interp - dchi_orig)) / factor

        # Restrict to kF*r < 15
        mask = kF * r_eval < 15
        max_dr = np.max(diff[mask]) if np.any(mask) else 999

        # MADE
        ref = np.abs(factor * dchi_orig[mask]) / NF
        made = np.sum(diff[mask]) / (np.sum(ref) + 1e-30) * 100

        results.append({"rs": rs, "max_dr": max_dr, "made_r": made})

    max_r = max(d["max_dr"] for d in results)
    mean_r = np.mean([d["max_dr"] for d in results])
    mean_made = np.mean([d["made_r"] for d in results])

    worst_p = max(
        ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"],
        key=lambda p: fits[p]["maxpct"] if fits[p]["popt"] is not None else 999,
    )
    wp = fits[worst_p]["maxpct"] if fits[worst_p]["popt"] is not None else 999

    return max_r, mean_r, mean_made, worst_p, wp, fits


# Form map candidates
PZ23 = "PZ[2/3]"
PZ23s = "PZ[2/3]\u221a"
mPZ23 = "mPZ[2/3]"
mPZ23s = "mPZ[2/3]\u221a"
PZ22s = "PZ[2/2]\u221a"
PZ22 = "Pade[2/2]"

candidates = {
    "your_best": {
        "alpha0": PZ23,
        "f0": PZ22s,
        "phi0": PZ23,
        "alpha1": PZ23,
        "f1": PZ22s,
        "phi1": PZ23s,
    },
    "all_PZ23": {
        "alpha0": PZ23,
        "f0": PZ23,
        "phi0": PZ23,
        "alpha1": PZ23,
        "f1": PZ23,
        "phi1": PZ23,
    },
    "all_PZ23sqrt": {
        "alpha0": PZ23s,
        "f0": PZ23s,
        "phi0": PZ23s,
        "alpha1": PZ23s,
        "f1": PZ23s,
        "phi1": PZ23s,
    },
    "all_mPZ23": {
        "alpha0": mPZ23,
        "f0": mPZ23,
        "phi0": mPZ23,
        "alpha1": mPZ23,
        "f1": mPZ23,
        "phi1": mPZ23,
    },
    "all_mPZ23sqrt": {
        "alpha0": mPZ23s,
        "f0": mPZ23s,
        "phi0": mPZ23s,
        "alpha1": mPZ23s,
        "f1": mPZ23s,
        "phi1": mPZ23s,
    },
    # Same form per physical quantity (alpha=alpha, f=f, phi=phi)
    "phys_A": {
        "alpha0": PZ23,
        "f0": PZ23s,
        "phi0": PZ23,
        "alpha1": PZ23,
        "f1": PZ23s,
        "phi1": PZ23,
    },
    "phys_B": {
        "alpha0": PZ23,
        "f0": mPZ23s,
        "phi0": PZ23,
        "alpha1": PZ23,
        "f1": mPZ23s,
        "phi1": PZ23,
    },
    "phys_C": {
        "alpha0": PZ23,
        "f0": PZ22s,
        "phi0": PZ23,
        "alpha1": PZ23,
        "f1": PZ22s,
        "phi1": PZ23,
    },
    "phys_D": {
        "alpha0": PZ23,
        "f0": mPZ23s,
        "phi0": PZ23s,
        "alpha1": PZ23,
        "f1": mPZ23s,
        "phi1": PZ23s,
    },
    "phys_E": {
        "alpha0": mPZ23,
        "f0": mPZ23s,
        "phi0": mPZ23,
        "alpha1": mPZ23,
        "f1": mPZ23s,
        "phi1": mPZ23,
    },
    "phys_F": {
        "alpha0": PZ23,
        "f0": PZ22s,
        "phi0": PZ23s,
        "alpha1": PZ23,
        "f1": PZ22s,
        "phi1": PZ23s,
    },
    # Pade[2/2] baseline
    "all_Pade22": {
        "alpha0": PZ22,
        "f0": PZ22,
        "phi0": PZ22,
        "alpha1": PZ22,
        "f1": PZ22,
        "phi1": PZ22,
    },
}

print(
    f"{'Name':20s} | {'max|dchi|':>10s} {'mean|dchi|':>10s} {'MADE%':>8s} | {'worst_param':>15s} | #coeff"
)
print("-" * 85)

ranking = []
for name, fm in candidates.items():
    t0 = time.time()
    try:
        max_r, mean_r, mean_made, worst_p, wp, fits = eval_form_map(
            fm, params_dict, rs_test, r_eval, orig_dchi
        )
        if max_r is None:
            print(f"{name:20s} | {'FAILED: ' + worst_p + ' fit failed':>50s}")
            continue
        ncoeff = sum(
            PARAMETRIC_FORMS[fm[p]][1]
            for p in ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]
        )
        dt = time.time() - t0
        print(
            f"{name:20s} | {max_r:10.6f} {mean_r:10.6f} {mean_made:7.3f}% | {worst_p:>7s}({wp:5.1f}%) | {ncoeff:3d}  ({dt:.1f}s)"
        )
        ranking.append((name, max_r, mean_r, mean_made, ncoeff, fm))
    except Exception as e:
        print(f"{name:20s} | FAILED: {e}")

print("\n=== RANKING by mean|dchi/NF| ===")
ranking.sort(key=lambda x: x[2])
for i, (name, max_r, mean_r, made, nc, fm) in enumerate(ranking):
    print(
        f"  {i + 1}. {name:20s}: max={max_r:.6f}, mean={mean_r:.6f}, MADE={made:.3f}%, {nc} coeffs"
    )

# Show best details
if ranking:
    best_name = ranking[0][0]
    best_fm = ranking[0][5]
    print(f"\nBest form map: {best_name}")
    print("form_map = {")
    for p in ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]:
        print(f'    "{p}": "{best_fm[p]}",')
    print("}")
