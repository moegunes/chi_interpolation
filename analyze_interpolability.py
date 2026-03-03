"""Analyze interpolability of the current smooth parameter curves."""

import sys

import numpy as np

sys.path.insert(0, "src")
from scipy.optimize import curve_fit

from utils.io import load_dict
from utils.physics import J_n_m_kFr, delta_C

params = load_dict("parameters")
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])
rs = np.array(rsl)
n = len(rs)

# Extract all 8 quantities
names6 = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]
data6 = np.array([params[r] for r in rsl])
B0_arr, B1_arr = [], []
for r_val in rsl:
    p = params[r_val]
    kF = (9 * np.pi / 4) ** (1 / 3) / r_val
    k0, k1 = 2 * np.pi * p[1], 2 * np.pi * p[4]
    J0 = J_n_m_kFr(0, k0, p[0], p[2], kF)
    J1 = J_n_m_kFr(0, k1, p[3], p[5], kF)
    J3 = J_n_m_kFr(1, k0, p[0], p[2], kF)
    J4 = J_n_m_kFr(1, k1, p[3], p[5], kF)
    M = np.array([[J3, J4], [J0, J1]])
    b = np.array([delta_C(1, r_val), delta_C(0, r_val)])
    B0, B1 = np.linalg.solve(M, b)
    B0_arr.append(B0)
    B1_arr.append(B1)
B0_arr = np.array(B0_arr)
B1_arr = np.array(B1_arr)

all_data = np.column_stack([data6, B0_arr, B1_arr])
all_names = names6 + ["B0", "B1"]


# ===== Candidate parametric forms =====
def f_denom1(x, a, b, c):
    """A / (rs**B + C)"""
    return a / (x**b + c)


def f_superpose(x, a, b, c, d):
    """A*rs**B + C*rs**D"""
    return a * x**b + c * x**d


def f_power(x, a, b, c):
    """a + b*rs^c (3 params)"""
    return a + b * x**c


def f_poly2(x, a, b, c):
    """a + b*rs + c*rs^2 (3 params)"""
    return a + b * x + c * x**2


def f_poly3(x, a, b, c, d):
    """a + b*rs + c*rs^2 + d*rs^3 (4 params)"""
    return a + b * x + c * x**2 + d * x**3


def f_pade11(x, a, b, c):
    """(a + b*rs)/(1 + c*rs) (3 params)"""
    return (a + b * x) / (1 + c * x)


def f_pade22(x, a, b, c, d, e):
    """(a + b*rs + c*rs^2)/(1 + d*rs + e*rs^2) (5 params)"""
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


def f_inv(x, a, b, c):
    """a + b/(rs + c) (3 params)"""
    return a + b / (x + c)


def f_sat(x, a, b, c):
    """a * (1 - exp(-b*rs)) + c (3 params)"""
    return a * (1 - np.exp(-b * x)) + c


def f_dblpow(x, a, b, c, d, e):
    """a + b*rs^c + d*rs^e (5 params)"""
    with np.errstate(invalid="ignore"):
        return a + b * np.abs(x) ** c + d * np.abs(x) ** e


forms = {
    "a+b*rs^c": (f_power, 3),
    "poly2": (f_poly2, 3),
    "poly3": (f_poly3, 4),
    "Pade[1/1]": (f_pade11, 3),
    "Pade[2/2]": (f_pade22, 5),
    "a+b/(rs+c)": (f_inv, 3),
    "saturating": (f_sat, 3),
    "a+b*rs^c+d*rs^e": (f_dblpow, 5),
    "denom": (f_denom1, 3),
    "superpose": (f_superpose, 4),
}

print("=" * 105)
print(
    f"{'Parameter':>12s} | {'Form':>20s} | {'ncoeff':>6s} | "
    f"{'MaxErr':>10s} | {'RMSE':>10s} | {'MaxErr%':>8s}"
)
print("=" * 105)

for j, name in enumerate(all_names):
    y = all_data[:, j]
    rng = np.max(y) - np.min(y)
    results = []
    for form_name, (func, ncoeff) in forms.items():
        try:
            popt, _ = curve_fit(func, rs, y, maxfev=50000)
            pred = func(rs, *popt)
            residuals = np.abs(pred - y)
            rmse = np.sqrt(np.mean(residuals**2))
            maxerr = np.max(residuals)
            maxpct = maxerr / (rng + 1e-30) * 100
            results.append((form_name, ncoeff, rmse, maxerr, maxpct, popt))
        except Exception:
            pass
    results.sort(key=lambda x: x[2])
    for rank, (form_name, ncoeff, rmse, maxerr, maxpct, popt) in enumerate(results[:3]):
        marker = "***" if rank == 0 else "   "
        lbl = name if rank == 0 else ""
        print(
            f"{lbl:>12s} {marker} {form_name:>20s} | {ncoeff:>6d} | "
            f"{maxerr:>10.6f} | {rmse:>10.6f} | {maxpct:>7.1f}%"
        )
    print("-" * 105)

print()
print("=== Detailed best-fit analysis (by RMSE) ===")
print()
total_coeffs = 0
for j, name in enumerate(all_names):
    y = all_data[:, j]
    rng = np.max(y) - np.min(y)
    best = None
    for form_name, (func, ncoeff) in forms.items():
        try:
            popt, _ = curve_fit(func, rs, y, maxfev=50000)
            pred = func(rs, *popt)
            rmse = np.sqrt(np.mean((pred - y) ** 2))
            maxerr = np.max(np.abs(pred - y))
            if best is None or rmse < best[1]:
                best = (form_name, rmse, maxerr, popt, ncoeff, func)
        except Exception:
            pass
    form_name, rmse, maxerr, popt, ncoeff, func = best
    maxpct = maxerr / (rng + 1e-30) * 100
    pred = func(rs, *popt)
    worst_idx = np.argmax(np.abs(pred - y))
    total_coeffs += ncoeff
    print(f"{name}: best={form_name} ({ncoeff} coeffs)")
    print(f"  RMSE={rmse:.6e}, MaxErr={maxerr:.6e} ({maxpct:.1f}% of range)")
    print(
        f"  Worst at rs={rs[worst_idx]:.2f} "
        f"(pred={pred[worst_idx]:.6f}, actual={y[worst_idx]:.6f})"
    )
    print(f"  Coefficients: {popt}")
    print()

print(
    f"Total meta-parameters (6 nonlinear only): {total_coeffs - (best[4] if name in ['B0', 'B1'] else 0)}"
)
print()

# Also check: if we restrict to rs >= 1.0 (where things are smoothest)
print("=" * 80)
print("=== Restricted analysis: rs >= 1.0 only (smooth regime) ===")
print("=" * 80)
mask = rs >= 1.0
rs_smooth = rs[mask]
for j, name in enumerate(all_names[:6]):  # only nonlinear params
    y = all_data[mask, j]
    rng = np.max(y) - np.min(y)
    results = []
    for form_name, (func, ncoeff) in forms.items():
        try:
            popt, _ = curve_fit(func, rs_smooth, y, maxfev=50000)
            pred = func(rs_smooth, *popt)
            rmse = np.sqrt(np.mean((pred - y) ** 2))
            maxerr = np.max(np.abs(pred - y))
            maxpct = maxerr / (rng + 1e-30) * 100
            results.append((form_name, ncoeff, rmse, maxerr, maxpct))
        except Exception:
            pass
    results.sort(key=lambda x: x[2])
    if results:
        best = results[0]
        print(
            f"  {name:>8s}: {best[0]:>20s} ({best[1]} coeffs), "
            f"RMSE={best[2]:.6e}, MaxErr={best[3]:.6e} ({best[4]:.1f}%)"
        )
