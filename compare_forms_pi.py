"""Compare all parametric forms for Pi(q) interpolation error.

For each form in PARAMETRIC_FORMS, fits the 6 parameters vs rs, reconstructs
Pi(q) from the interpolated parameters, and compares against exact Pi(q).
Reports worst max|δΠ(q)|/NF and mean MADE for rs ∈ [0.5, 10.0].
"""

import pickle
import sys

import numpy as np

sys.path.insert(0, "src")

from input import q, r
from optimization.production import get_pi_interp
from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import get_gas_params, get_piq
from visualization.pp import (
    PARAMETRIC_FORMS,
    fit_all_parameters,
    get_interpolated_params,
)

# Load fitted parameters
with open("parameters.pkl", "rb") as f:
    params_dict = pickle.load(f)

# rs range for testing
rsl = sorted([k for k in params_dict if isinstance(k, (int, float)) and k != "model"])
rs_test = [rs for rs in rsl if 0.5 <= rs <= 10.0]
print(f"Testing {len(rs_test)} rs values in [0.5, 10.0]")
print()

# Forms to test
forms_to_test = [
    "mPZ[2/3]√",
    "mPZ[2/3]",
    "PZ[2/3]√",
    "PZ[2/3]",
    "PZ[2/2]√",
    "Pade[2/2]",
    "PZ[2/1]√",
    "PZ[1/1]√",
    "Pade[1/1]",
    "two Pade[1/1]",
    "two Pade[1/1]√",
]

print(
    f"{'Form':<20s}  {'#p':>3s}  {'worst max|δΠ(q)|/NF':>20s}  {'worst_rs':>8s}  {'mean MADE%':>10s}  {'worst MADE%':>11s}  {'worst_MADE_rs':>13s}  {'status':>8s}"
)
print("-" * 110)

results = []

for form_name in forms_to_test:
    if form_name not in PARAMETRIC_FORMS:
        print(f"{form_name:<20s}  SKIPPED (not in PARAMETRIC_FORMS)")
        continue

    ncoeff = PARAMETRIC_FORMS[form_name][1]

    try:
        fits = fit_all_parameters(params_dict, form_name)

        # Check if any parameter fit failed
        failed_params = []
        for pname in ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]:
            if fits[pname].get("popt") is None:
                failed_params.append(pname)

        if failed_params:
            print(
                f"{form_name:<20s}  {ncoeff:3d}  FAILED (params: {', '.join(failed_params)})"
            )
            continue

        # Evaluate Pi(q) error for each rs
        max_diffs_q = []
        made_vals = []
        rs_results = []

        for rs in rs_test:
            kF, n0, NF = get_gas_params(rs)

            # Exact Pi(q)
            piq_exact = get_piq(q, rs)

            # Interpolated Pi(r) → FT → Pi(q)
            params_interp = get_interpolated_params(rs, fits)
            temp_dict = {rs: params_interp, "model": params_dict["model"]}
            pi_interp_r = get_pi_interp(r, q, temp_dict, rs)
            FT_q, FT_piq = chi_q_from_chi_r_fast(r, pi_interp_r, qlist=None)

            # Error: |δΠ(q)| / NF
            q_mask = q < 6.0 * kF
            diff_q = np.abs(FT_piq[q_mask] - piq_exact[q_mask]) / NF
            max_diff = np.max(diff_q)

            # MADE
            made = (
                np.sum(diff_q) / (np.sum(np.abs(piq_exact[q_mask])) / NF + 1e-30) * 100
            )

            max_diffs_q.append(max_diff)
            made_vals.append(made)
            rs_results.append({"rs": rs, "max_diff_q": max_diff, "MADE_%": made})

        worst_idx = np.argmax(max_diffs_q)
        worst_made_idx = np.argmax(made_vals)

        print(
            f"{form_name:<20s}  {ncoeff:3d}  {max_diffs_q[worst_idx]:20.6f}  "
            f"{rs_test[worst_idx]:8.2f}  {np.mean(made_vals):10.4f}  "
            f"{made_vals[worst_made_idx]:11.4f}  {rs_test[worst_made_idx]:13.2f}  "
            f"{'OK':>8s}"
        )

        results.append(
            {
                "form": form_name,
                "ncoeff": ncoeff,
                "worst_max_diff_q": max_diffs_q[worst_idx],
                "worst_rs": rs_test[worst_idx],
                "mean_MADE": np.mean(made_vals),
                "worst_MADE": made_vals[worst_made_idx],
                "worst_MADE_rs": rs_test[worst_made_idx],
                "per_rs": rs_results,
            }
        )

    except Exception as e:
        print(f"{form_name:<20s}  {ncoeff:3d}  ERROR: {e}")

# Summary: rank by worst max|δΠ(q)|/NF
print()
print("=" * 80)
print("RANKING by worst max|δΠ(q)|/NF (lower is better):")
print("=" * 80)
results_sorted = sorted(results, key=lambda x: x["worst_max_diff_q"])
for i, res in enumerate(results_sorted):
    print(
        f"  {i + 1}. {res['form']:<20s} ({res['ncoeff']}p)  "
        f"worst={res['worst_max_diff_q']:.6f} @ rs={res['worst_rs']:.2f}  "
        f"mean_MADE={res['mean_MADE']:.4f}%  worst_MADE={res['worst_MADE']:.4f}%"
    )

print()
print("RANKING by mean MADE% (lower is better):")
print("=" * 80)
results_sorted_made = sorted(results, key=lambda x: x["mean_MADE"])
for i, res in enumerate(results_sorted_made):
    print(
        f"  {i + 1}. {res['form']:<20s} ({res['ncoeff']}p)  "
        f"mean_MADE={res['mean_MADE']:.4f}%  worst_MADE={res['worst_MADE']:.4f}%  "
        f"worst_max={res['worst_max_diff_q']:.6f}"
    )

# Per-rs breakdown for the best form
if results_sorted:
    best = results_sorted[0]
    print()
    print(f"Per-rs breakdown for best form: {best['form']}")
    print(f"  {'rs':>5s}  {'max|δΠ(q)|/NF':>15s}  {'MADE%':>10s}")
    for r_res in best["per_rs"]:
        print(
            f"  {r_res['rs']:5.2f}  {r_res['max_diff_q']:15.6f}  {r_res['MADE_%']:10.4f}"
        )
