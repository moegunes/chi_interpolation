"""Investigate phi1 failure with PZ[2/3] and coefficient count tradeoff."""

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

# Why does PZ[2/3] fail for phi1?
y_phi1 = data6[:, 5]
form_name = "PZ[2/3]"
func, nc = PARAMETRIC_FORMS[form_name]

try:
    popt, _ = curve_fit(func, rs_arr, y_phi1, maxfev=50000)
    pred = func(rs_arr, *popt)
    rng = y_phi1.max() - y_phi1.min()
    maxerr = np.max(np.abs(pred - y_phi1)) / rng * 100
    print(f"PZ[2/3] phi1: maxerr={maxerr:.2f}%, popt={popt}")
except Exception as e:
    print(f"PZ[2/3] phi1: FAILED - {e}")

# Try with multiple random initial guesses
np.random.seed(42)
rng_val = y_phi1.max() - y_phi1.min()
for trial in range(5):
    try:
        p0 = np.random.randn(nc) * 0.1
        popt, _ = curve_fit(func, rs_arr, y_phi1, p0=p0, maxfev=100000)
        pred = func(rs_arr, *popt)
        maxerr = np.max(np.abs(pred - y_phi1)) / rng_val * 100
        print(f"  trial {trial}: maxerr={maxerr:.2f}%")
    except Exception as e:
        print(f"  trial {trial}: FAILED - {e}")

# Coefficient count vs accuracy summary
print()
print("=== Coefficient count vs accuracy summary ===")
forms_to_check = [
    "Pade[2/2]",
    "PZ[2/3]\u221a",
    "PZ[2/3]",
    "mPZ[2/3]",
    "mPZ[2/3]\u221a",
]
param_names = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]

for form in forms_to_check:
    f, nc = PARAMETRIC_FORMS[form]
    errs = []
    for i, pname in enumerate(param_names):
        y = data6[:, i]
        try:
            popt, _ = curve_fit(f, rs_arr, y, maxfev=50000)
            pred = f(rs_arr, *popt)
            rng = y.max() - y.min()
            maxpct = np.max(np.abs(pred - y)) / (rng + 1e-30) * 100
            errs.append(f"{maxpct:.1f}%")
        except:
            errs.append("FAIL")
    total_c = nc * 6
    print(f"{form:15s} ({nc}p x6 = {total_c:2d}): {errs}")
