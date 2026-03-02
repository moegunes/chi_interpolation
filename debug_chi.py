"""Debug chi error and show actual situation."""

import sys

import numpy as np

sys.path.insert(0, "src")
from input import q, r
from optimization.production import get_chi_interp
from utils.fourier import chi_q_from_chi_r_fast
from utils.io import load_dict
from utils.utils_chi import get_chi, get_gas_params

params = load_dict("parameters")
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])
data = np.array([params[rs] for rs in rsl])

print("Current fit quality (chi error at each rs):")
print("=" * 60)

for rs in [0.5, 1.0, 2.0, 4.0, 8.0]:
    if rs not in rsl:
        continue
    kF, n0, NF = get_gas_params(rs)

    # Target chi(q)
    chi_target = get_chi(q, rs)

    # Our model chi(r) -> chi(q)
    chi_r_model = get_chi_interp(r, q, params, rs)
    _, chi_q_model = chi_q_from_chi_r_fast(r, chi_r_model, qlist=q)

    # Error in interesting q range (0.5 to 4 kF)
    q_mask = (q > 0.5 * kF) & (q < 4 * kF)
    diff = np.abs(chi_q_model[q_mask] - chi_target[q_mask]) / NF
    rel_diff = diff / (np.max(np.abs(chi_target[q_mask])) / NF)

    print(
        f"rs={rs:.1f}: max |Δχ(q)|/NF = {np.max(diff):.4e}, rel = {np.max(rel_diff) * 100:.2f}%"
    )

print("\n" + "=" * 60)
print("Interpolability analysis (Padé[2/2] parameter fit error):")
print("=" * 60)

from scipy.optimize import curve_fit


def pade22(x, a, b, c, d, e):
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


rs_arr = np.array(rsl)
labels = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]

for i, lab in enumerate(labels):
    y = data[:, i]
    rng = y.max() - y.min()
    try:
        popt, _ = curve_fit(pade22, rs_arr, y, maxfev=50000)
        pred = pade22(rs_arr, *popt)
        maxerr_pct = np.max(np.abs(pred - y)) / rng * 100
        worst_idx = np.argmax(np.abs(pred - y))
        print(f"{lab:8s}: MaxErr = {maxerr_pct:5.1f}% at rs={rsl[worst_idx]:.2f}")
    except:
        print(f"{lab:8s}: FAILED")

# The key question: What parameter interpolation error is acceptable?
print("\n" + "=" * 60)
print("ANALYSIS: Translating parameter error to chi error")
print("=" * 60)

# Pick rs=1.0 where phi0 problem is worst
rs_test = 1.0
if rs_test in rsl:
    idx = rsl.index(rs_test)
    orig_params = data[idx].copy()
    kF, n0, NF = get_gas_params(rs_test)

    # Baseline chi error
    chi_r_orig = get_chi_interp(r, q, params, rs_test)
    _, chi_q_orig = chi_q_from_chi_r_fast(r, chi_r_orig, qlist=q)
    chi_target = get_chi(q, rs_test)

    q_mask = (q > 0.5 * kF) & (q < 4 * kF)
    baseline_err = np.max(np.abs(chi_q_orig[q_mask] - chi_target[q_mask]) / NF)

    print(f"\nAt rs={rs_test}, baseline chi error = {baseline_err:.4e}")

    # Test: perturb phi0 by various amounts
    print("\nEffect of phi0 perturbation on chi error:")
    for dphi in [0.01, 0.05, 0.1, 0.2]:
        test_params = orig_params.copy()
        test_params[2] += dphi  # perturb phi0

        temp_dict = {rs_test: test_params, "model": params["model"]}
        chi_r_test = get_chi_interp(r, q, temp_dict, rs_test)
        _, chi_q_test = chi_q_from_chi_r_fast(r, chi_r_test, qlist=q)

        test_err = np.max(np.abs(chi_q_test[q_mask] - chi_target[q_mask]) / NF)
        delta_err = test_err - baseline_err

        print(
            f"  Δphi0 = {dphi:.2f} rad: chi error = {test_err:.4e} (Δ = {delta_err:+.4e})"
        )

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("""
Based on analysis, the key issue is phi0 with 68.8% Padé[2/2] error.
The phi0 values show a non-monotonic "dip" around rs ~ 0.8-1.2.

OPTIONS:
1. HIGHER LAMBDA: Increase lambda_smooth from 10 to 100 or 1000
   - Pros: Easy to test
   - Cons: May hurt fit quality significantly
   
2. ALTERNATIVE BRANCHES: Search with many random starts at rs~1
   - Pros: May find smoother trajectory  
   - Cons: Expensive, may not exist
   
3. PIECEWISE PADÉ: Use different Padé for rs<1.5 vs rs>1.5 for phi0
   - Pros: Could capture the dip
   - Cons: Discontinuity at junction
   
4. DIRECT PADÉ FIT: Re-fit all rs forcing params = Padé(rs)
   - Pros: Guarantees interpolability
   - Cons: May significantly degrade chi fit quality

5. ACCEPT PHI0 AS LOOKUP: Use Padé for 5 params, spline for phi0
   - Pros: ~perfect interpolability for 5/6 params
   - Cons: Need lookup table for phi0
""")
