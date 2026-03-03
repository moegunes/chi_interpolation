"""Fit mPZ[2/3] to all 6 parameters and save production coefficients."""

import json
import sys

import numpy as np

sys.path.insert(0, "src")
from scipy.optimize import curve_fit

from utils.io import load_dict
from visualization.pp import PARAMETRIC_FORMS

params_dict = load_dict("parameters")
rsl = sorted(
    [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
)
rs_arr = np.array(rsl)
data6 = np.array([params_dict[rs] for rs in rsl])

form_name = "mPZ[2/3]"
func, nc = PARAMETRIC_FORMS[form_name]

param_names = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]

coefficients = {}
print(f"Fitting {form_name} ({nc} params each) to all 6 parameters\n")
print(f"{'param':8s} | {'maxerr%':>8s} | {'RMSE':>10s} | coefficients")
print("-" * 100)

for i, pname in enumerate(param_names):
    y = data6[:, i]
    rng = y.max() - y.min()

    # Try multiple initial guesses to get best fit
    best_popt = None
    best_maxpct = 999
    for trial in range(20):
        try:
            if trial == 0:
                popt, _ = curve_fit(func, rs_arr, y, maxfev=100000)
            else:
                np.random.seed(trial * 13 + i * 7)
                p0 = np.random.randn(nc) * 0.5
                popt, _ = curve_fit(func, rs_arr, y, p0=p0, maxfev=100000)
            pred = func(rs_arr, *popt)
            maxpct = np.max(np.abs(pred - y)) / (rng + 1e-30) * 100
            if maxpct < best_maxpct:
                best_maxpct = maxpct
                best_popt = popt
        except:
            pass

    rmse = np.sqrt(np.mean((func(rs_arr, *best_popt) - y) ** 2))
    coefficients[pname] = best_popt.tolist()

    coeff_str = ", ".join(f"{c:.10e}" for c in best_popt)
    print(f"{pname:8s} | {best_maxpct:7.2f}% | {rmse:10.4e} | [{coeff_str}]")

# Save as JSON for the production module
output = {
    "form": form_name,
    "description": "g + (a + b*rs + c*rs^2 + h*rs^3) / (1 + d*rs + e*rs^2 + f*rs^3)",
    "n_coefficients_per_param": nc,
    "total_coefficients": nc * 6,
    "param_names": param_names,
    "coefficients": coefficients,
    "rs_range_fitted": [float(rs_arr.min()), float(rs_arr.max())],
    "rs_range_recommended": [0.5, 10.0],
}

with open("interpolation_coefficients.json", "w") as fp:
    json.dump(output, fp, indent=2)
print("\nSaved to interpolation_coefficients.json")
print(f"Total meta-parameters: {nc * 6}")

# Also save as numpy for convenience
np.save("interpolation_coefficients.npy", output)
print("Saved to interpolation_coefficients.npy")

# Verify round-trip: reconstruct delta_chi at a test point
print("\n=== Verification ===")
from optimization.models import delta_chi as delta_chi_model
from utils.utils_chi import get_gas_params

for rs_test in [1.0, 3.0, 5.0, 8.0]:
    # Reconstruct params from coefficients
    params_interp = np.array([func(rs_test, *coefficients[p]) for p in param_names])
    params_orig = params_dict[rs_test]

    kF, n0, NF = get_gas_params(rs_test)
    factor = -6 * np.pi * n0 * NF

    r_test = np.linspace(0.01, 50, 3000)
    dchi_orig = delta_chi_model(r_test, rs=rs_test, params=params_orig)
    dchi_interp = delta_chi_model(r_test, rs=rs_test, params=params_interp)

    mask = kF * r_test < 15
    diff = np.abs(factor * (dchi_interp - dchi_orig))
    max_err = np.max(diff[mask]) / NF

    print(
        f"  rs={rs_test:4.1f}: max|dchi|/NF = {max_err:.6f}, params_diff = {np.max(np.abs(params_interp - params_orig)):.6f}"
    )
