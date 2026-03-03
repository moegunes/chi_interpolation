"""Explore strategies for better parameter interpolability."""

import sys

import numpy as np

sys.path.insert(0, "src")

from scipy.optimize import curve_fit

from input import q, r
from utils.fourier import chi_q_from_chi_r_fast
from utils.io import load_dict
from utils.utils_chi import get_chi, get_chi02, get_gas_params

# Load current parameters
params = load_dict("parameters")
model = params["model"]
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])
rs_arr = np.array(rsl)
data = np.array([params[rs] for rs in rsl])


# Padé[2/2] form
def pade22(x, a, b, c, d, e):
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


# ============================================================
# STRATEGY 1: Higher lambda_smooth test
# ============================================================
print("=" * 70)
print("STRATEGY 1: Test higher lambda_smooth values")
print("=" * 70)


def compute_chi_error(rs, params_6):
    """Compute chi(q) fit error for given parameters."""
    kF, n0, NF = get_gas_params(rs)
    chi_target = get_chi(q, rs)

    # Get chi from model
    temp_dict = {rs: params_6, "model": model}
    from optimization.production import get_chi_interp

    chi_pred = get_chi_interp(r, q, temp_dict, rs)
    _, chi_q_pred = chi_q_from_chi_r_fast(r, chi_pred, qlist=q)

    # Relative error
    rel_err = np.max(np.abs(chi_q_pred - chi_target)) / np.max(np.abs(chi_target)) * 100
    return rel_err


print("\nCurrent chi(q) errors across rs:")
chi_errors = []
for i, rs in enumerate(rsl[:10]):  # First 10 for speed
    err = compute_chi_error(rs, data[i])
    chi_errors.append(err)
    print(f"  rs={rs:.2f}: chi(q) error = {err:.2f}%")

# ============================================================
# STRATEGY 2: Search for alternative branches at problematic rs
# ============================================================
print("\n" + "=" * 70)
print("STRATEGY 2: Search for alternative branches at phi0's problematic region")
print("=" * 70)

from optimization.models import delta_chi


def fit_cost(params_6, rs):
    """Compute fitting cost for parameters at given rs."""
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    chiR = get_chi(q, rs)
    chi0R = get_chi02(q, rs)
    dchi = -(chi0R - chiR) / factor

    # Fit region
    i0 = np.argmin(np.abs(kF * r - 0))
    i1 = np.argmin(np.abs(kF * r - 4))
    yf = dchi[i0:i1]

    # Model prediction
    pred = delta_chi(r[i0:i1], rs, params_6)
    return np.sqrt(np.mean((pred - yf) ** 2))


# At rs=1.0 and 1.1 where phi0 is problematic, search for better branches
test_rs_values = [0.8, 0.9, 1.0, 1.1, 1.2]
print("\nTrying to find alternative branches at problematic rs:")

for test_rs in test_rs_values:
    if test_rs not in rsl:
        continue
    idx = rsl.index(test_rs)
    current = data[idx]
    current_cost = fit_cost(current, test_rs)

    # Try interpolated guess from neighbors
    if idx > 0 and idx < len(rsl) - 1:
        interp_guess = 0.5 * (data[idx - 1] + data[idx + 1])
        interp_cost = fit_cost(interp_guess, test_rs)
    else:
        interp_guess = current
        interp_cost = current_cost

    print(f"\n  rs={test_rs:.2f}:")
    print(
        f"    Current params: alpha0={current[0]:.3f}, f0={current[1]:.4f}, phi0={current[2]:.4f}"
    )
    print(f"    Current cost: {current_cost:.4e}")
    print(
        f"    Interp params: alpha0={interp_guess[0]:.3f}, f0={interp_guess[1]:.4f}, phi0={interp_guess[2]:.4f}"
    )
    print(f"    Interp cost:  {interp_cost:.4e}")
    print(
        f"    Cost increase: {(interp_cost - current_cost) / current_cost * 100:.1f}%"
    )

# ============================================================
# STRATEGY 3: Direct parametric fitting
# ============================================================
print("\n" + "=" * 70)
print("STRATEGY 3: Direct parametric fitting (constrain params to follow Padé)")
print("=" * 70)

# Fit Padé to all parameters
print("\nBest Padé[2/2] fits to current parameters:")
pade_coeffs = {}
for i, lab in enumerate(["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]):
    y = data[:, i]
    try:
        popt, _ = curve_fit(pade22, rs_arr, y, maxfev=50000)
        pred = pade22(rs_arr, *popt)
        maxerr_pct = np.max(np.abs(pred - y)) / (y.max() - y.min()) * 100
        pade_coeffs[lab] = popt
        print(f"  {lab}: MaxErr = {maxerr_pct:.1f}%")
    except:
        print(f"  {lab}: FAILED")

# What if we force parameters to follow Padé and re-optimize the Padé coefficients?
print("\nEstimated chi(q) error if params follow perfect Padé[2/2]:")


def pade_params_from_coeffs(rs, all_coeffs):
    """Get 6 params at given rs from Padé coefficients."""
    p = []
    for lab in ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]:
        p.append(pade22(rs, *all_coeffs[lab]))
    return np.array(p)


# Estimate effect on chi
print("  (Computing for a few rs values...)")
for test_rs in [0.5, 1.0, 2.0, 4.0, 8.0]:
    if test_rs in rsl:
        idx = rsl.index(test_rs)
        orig_params = data[idx]
        pade_params = pade_params_from_coeffs(test_rs, pade_coeffs)

        orig_err = compute_chi_error(test_rs, orig_params)
        pade_err = compute_chi_error(test_rs, pade_params)

        print(
            f"    rs={test_rs:.1f}: original={orig_err:.2f}%, Padé-constrained={pade_err:.2f}%"
        )

# ============================================================
# ANALYSIS: What would "acceptable" error look like?
# ============================================================
print("\n" + "=" * 70)
print("TARGET: What interpolability do we need?")
print("=" * 70)
print("""
If we want χ(q) errors < 1%, parameter interpolation errors should be ~0.1-1%.
Current phi0 at 68.8% is WAY too high.

Options ranked by feasibility:
1. Higher lambda (100-1000): Quick test, may sacrifice fit quality
2. Multi-branch search: Computationally expensive but might find smoother solution
3. Direct Padé constraint: Re-fit all rs simultaneously forcing P(rs) = Padé form
4. Per-parameter forms: Use different forms for phi0 vs others

Let's test option 3 with a toy implementation...
""")

# ============================================================
# OPTION 3 PROTOTYPE: Direct Padé-constrained fitting
# ============================================================
print("=" * 70)
print("PROTOTYPE: Direct Padé-constrained global fit")
print("=" * 70)

# This would jointly optimize:
# - 5 Padé coefficients for each of 6 parameters = 30 total coefficients
# - Minimize: sum over rs of chi(q) fitting error
#
# Parameters at each rs are NOT free - they're computed from Padé(rs)

# Count parameters
print(f"""
Standard approach: 6 params × {len(rsl)} rs = {6 * len(rsl)} free parameters
Padé-constrained:  5 coeffs × 6 params = 30 total parameters

Parameter reduction: {6 * len(rsl)} → 30 ({30 / (6 * len(rsl)) * 100:.1f}% of original)

This massive reduction forces interpolability by construction!
""")
