"""Test higher lambda_smooth values to improve interpolability."""

import sys

import numpy as np

sys.path.insert(0, "src")

from scipy.optimize import curve_fit

from utils.io import load_dict, write_dict


# Padé[2/2]
def pade22(x, a, b, c, d, e):
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


def compute_interpolability(params_dict):
    """Compute Padé[2/2] max error for all 6 params."""
    rsl = sorted(
        [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
    )
    rs_arr = np.array(rsl)
    data = np.array([params_dict[rs] for rs in rsl])

    results = {}
    labels = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]
    for i, lab in enumerate(labels):
        y = data[:, i]
        rng = y.max() - y.min()
        try:
            popt, _ = curve_fit(pade22, rs_arr, y, maxfev=50000)
            pred = pade22(rs_arr, *popt)
            results[lab] = np.max(np.abs(pred - y)) / (rng + 1e-30) * 100
        except:
            results[lab] = 999.0
    return results


# Load current (lambda=10) parameters
params_10 = load_dict("parameters")
print("Current parameters (lambda_smooth=10):")
interp_10 = compute_interpolability(params_10)
for k, v in interp_10.items():
    print(f"  {k:8s}: {v:.1f}%")
print(f"  Max: {max(interp_10.values()):.1f}%")

print("\n" + "=" * 70)
print("Testing higher lambda_smooth values...")
print("=" * 70)

# Test lambda = 50, 100, 500
from input import q, r
from optimization.fitting import _global_smooth_refit

rsl = sorted(
    [k for k in params_10.keys() if isinstance(k, (float, int)) and k != "model"]
)
model = params_10["model"]

for lam in [50, 100, 500]:
    print(f"\n--- Testing lambda_smooth = {lam} ---")

    # Start from current params_10
    params_test = {rs: params_10[rs].copy() for rs in rsl}
    params_test["model"] = model

    # Run global smooth refit with higher lambda
    params_result = _global_smooth_refit(
        rsl, q, r, model, params_test, lambda_smooth=lam, max_iter=2000
    )

    # Compute new interpolability
    interp_new = compute_interpolability(params_result)

    print("\nInterpolability after refit:")
    for k, v in interp_new.items():
        old = interp_10[k]
        print(f"  {k:8s}: {v:.1f}% (was {old:.1f}%)")
    print(
        f"  Max: {max(interp_new.values()):.1f}% (was {max(interp_10.values()):.1f}%)"
    )

    # Save if improved
    if max(interp_new.values()) < max(interp_10.values()):
        save_name = f"parameters_lambda{lam}"
        write_dict(params_result, save_name)
        print(f"  Saved as {save_name}")
