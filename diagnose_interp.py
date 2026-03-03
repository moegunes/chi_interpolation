"""Diagnose interpolability issues and explore solutions."""

import sys

import numpy as np

sys.path.insert(0, "src")
from scipy.optimize import curve_fit

from utils.io import load_dict

params = load_dict("parameters")
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])
rs_arr = np.array(rsl)
data = np.array([params[rs] for rs in rsl])


# Padé[2/2] form
def pade22(x, a, b, c, d, e):
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


labels = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]
print("Current interpolability (Padé[2/2]):")
print("=" * 60)
for i, lab in enumerate(labels):
    y = data[:, i]
    rng = y.max() - y.min()
    try:
        popt, _ = curve_fit(pade22, rs_arr, y, maxfev=50000)
        pred = pade22(rs_arr, *popt)
        maxerr = np.max(np.abs(pred - y))
        maxpct = maxerr / rng * 100
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        worst_idx = np.argmax(np.abs(pred - y))
        print(
            f"{lab:8s}: MaxErr = {maxpct:5.1f}% at rs={rsl[worst_idx]:.2f}, RMSE = {rmse:.4e}"
        )
    except Exception as e:
        print(f"{lab:8s}: FAILED - {e}")

# Identify jumps
print("\n" + "=" * 60)
print("Parameter jumps (> 5% of range):")
for i, lab in enumerate(labels):
    y = data[:, i]
    rng = y.max() - y.min()
    diffs = np.abs(np.diff(y))
    for j in range(len(diffs)):
        pct = diffs[j] / rng * 100
        if pct > 5:
            print(
                f"  {lab}: rs {rsl[j]:.2f} -> {rsl[j + 1]:.2f}: jump = {diffs[j]:.4f} ({pct:.1f}%)"
            )

# What are the fit costs at each rs?
print("\n" + "=" * 60)
print("Analyzing fit quality vs parameter behavior tradeoff...")

from input import q, r

# Test a few rs values to see if there are alternative branches
test_rs = [0.6, 0.8, 1.0, 1.2]
print("\nSearching for alternative branches at select rs values:")
for rs in test_rs:
    if rs not in rsl:
        continue
    idx = rsl.index(rs)
    current_params = data[idx]

    # Try different initial guesses
    from optimization.models import chi_model_hybrid

    best_cost = float("inf")
    best_params = None

    # Current fit quality
    from utils.utils_chi import get_chi

    chi_target = get_chi(q, rs)
    chi_pred = chi_model_hybrid(r, rs, current_params)
    from utils.fourier import chi_q_from_chi_r_fast

    _, chi_q_pred = chi_q_from_chi_r_fast(r, chi_pred, qlist=q)

    current_cost = np.sqrt(np.mean((chi_q_pred - chi_target) ** 2))
    print(
        f"  rs={rs}: current cost = {current_cost:.4e}, params = {current_params[:3]}"
    )
