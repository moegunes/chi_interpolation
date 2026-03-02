"""Direct Padé-constrained fitting: Optimize Padé coefficients to minimize chi error."""

import sys

import numpy as np

sys.path.insert(0, "src")

from scipy.optimize import curve_fit, minimize

from input import q, r
from utils.io import load_dict, write_dict
from utils.physics import delta_C as _delta_C
from utils.utils_chi import get_chi, get_chi02, get_gas_params


# Padé[2/2]: (a + b*x + c*x^2) / (1 + d*x + e*x^2)
def pade22(x, coeffs):
    a, b, c, d, e = coeffs
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


# Load current parameters as starting point for Padé coefficients
params = load_dict("parameters")
model = params["model"]
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])
rs_arr = np.array(rsl)
data = np.array([params[rs] for rs in rsl])

# Initial Padé coefficients from curve_fit to current data
print("Fitting initial Padé coefficients from current parameters...")
labels = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]
init_coeffs = {}


def pade22_fit(x, a, b, c, d, e):
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


for i, lab in enumerate(labels):
    y = data[:, i]
    try:
        popt, _ = curve_fit(pade22_fit, rs_arr, y, maxfev=50000)
        init_coeffs[lab] = popt.copy()
        pred = pade22(rs_arr, popt)
        maxerr = np.max(np.abs(pred - y)) / (y.max() - y.min()) * 100
        print(f"  {lab}: MaxErr = {maxerr:.1f}%")
    except Exception:
        print(f"  {lab}: FAILED - using quadratic init")
        # Fallback: simple polynomial fit then map to Padé
        from numpy.polynomial import polynomial as P

        c_poly = P.polyfit(rs_arr, y, 2)
        init_coeffs[lab] = np.array([c_poly[0], c_poly[1], c_poly[2], 0.0, 0.0])

# Flatten initial coefficients: [alpha0_a, alpha0_b, ..., phi1_e]
x0 = np.concatenate([init_coeffs[lab] for lab in labels])
print(f"\nTotal coefficients: {len(x0)} ({len(labels)} params × 5 Padé coeffs)")

# Precompute target data for all rs
print("\nPrecomputing target data...")
fit_data = []  # (rs, kF, y_fit, r_fit, delta_C1, delta_C0)

for rs_val in rsl:
    kF, n0, NF = get_gas_params(rs_val)
    factor = -6 * np.pi * n0 * NF
    chiR = get_chi(q, rs_val)
    chi0R = get_chi02(q, rs_val)
    dchi = -(chi0R - chiR) / factor

    # Fit region
    i0 = np.argmin(np.abs(kF * r - 0))
    i1 = np.argmin(np.abs(kF * r - 4))
    rf = r[i0:i1]
    yf = dchi[i0:i1]
    step = max(1, len(rf) // 200)  # subsample for speed

    fit_data.append(
        {
            "rs": rs_val,
            "kF": kF,
            "r_fit": rf[::step],
            "kFr": kF * rf[::step],
            "y_fit": yf[::step],
            "dC1": _delta_C(1, rs_val),
            "dC0": _delta_C(0, rs_val),
            "inv_kF3": 1.0 / kF**3,
            "inv_kF5": 1.0 / kF**5,
        }
    )


def objective(x):
    """Total chi fitting error with Padé-constrained parameters."""
    # Unpack Padé coefficients
    coeffs = {}
    for i, lab in enumerate(labels):
        coeffs[lab] = x[i * 5 : (i + 1) * 5]

    total_cost = 0.0

    for fd in fit_data:
        rs = fd["rs"]
        kFr = fd["kFr"]
        yf = fd["y_fit"]

        # Compute params from Padé at this rs
        a0 = pade22(rs, coeffs["alpha0"])
        f0 = pade22(rs, coeffs["f0"])
        ph0 = pade22(rs, coeffs["phi0"])
        a1 = pade22(rs, coeffs["alpha1"])
        f1 = pade22(rs, coeffs["f1"])
        ph1 = pade22(rs, coeffs["phi1"])

        # Enforce bounds
        a0 = max(1e-4, min(20.0, a0))
        a1 = max(1e-4, min(20.0, a1))
        f0 = max(0.02, min(3.0, f0))
        f1 = max(0.02, min(3.0, f1))

        # Compute model prediction
        k0 = 2.0 * np.pi * f0
        k1 = 2.0 * np.pi * f1
        e0 = np.exp(1j * ph0)
        e1 = np.exp(1j * ph1)
        z0 = a0 - 1j * k0
        z1 = a1 - 1j * k1

        # B0, B1 from constraints
        J0 = 2.0 * np.real(e0 / z0**3) * fd["inv_kF3"]
        J1 = 2.0 * np.real(e1 / z1**3) * fd["inv_kF3"]
        J3 = 24.0 * np.real(e0 / z0**5) * fd["inv_kF5"]
        J4 = 24.0 * np.real(e1 / z1**5) * fd["inv_kF5"]
        det = J3 * J1 - J4 * J0

        if abs(det) < 1e-30:
            return 1e30

        B0 = (fd["dC1"] * J1 - fd["dC0"] * J4) / det
        B1 = (fd["dC0"] * J3 - fd["dC1"] * J0) / det

        # Model
        pred = B0 * np.exp(-a0 * kFr) * np.cos(k0 * kFr + ph0) + B1 * np.exp(
            -a1 * kFr
        ) * np.cos(k1 * kFr + ph1)

        total_cost += np.sum((pred - yf) ** 2)

    return total_cost


# Compute initial cost
init_cost = objective(x0)
print(f"\nInitial cost: {init_cost:.4e}")

# Optimize
print("\nOptimizing Padé coefficients (L-BFGS-B)...")
print("This guarantees interpolability by construction!")

result = minimize(
    objective, x0, method="L-BFGS-B", options={"maxiter": 2000, "disp": True}
)

print("\nOptimization finished:")
print(f"  Final cost: {result.fun:.4e} (was {init_cost:.4e})")
print(f"  Cost ratio: {result.fun / init_cost:.2f}x")

# Extract final Padé coefficients and compute parameters
final_coeffs = {}
for i, lab in enumerate(labels):
    final_coeffs[lab] = result.x[i * 5 : (i + 1) * 5]

# Build new params_dict from Padé
params_pade = {"model": model}
print("\nNew interpolability (perfect by construction):")
for lab in labels:
    y_orig = data[:, labels.index(lab)]
    y_new = np.array([pade22(rs, final_coeffs[lab]) for rs in rsl])
    err = np.max(np.abs(y_new - y_orig)) / (y_orig.max() - y_orig.min()) * 100
    print(f"  {lab}: deviation from original = {err:.1f}%")

for i, rs in enumerate(rsl):
    params_pade[rs] = np.array(
        [
            pade22(rs, final_coeffs["alpha0"]),
            pade22(rs, final_coeffs["f0"]),
            pade22(rs, final_coeffs["phi0"]),
            pade22(rs, final_coeffs["alpha1"]),
            pade22(rs, final_coeffs["f1"]),
            pade22(rs, final_coeffs["phi1"]),
        ]
    )

# Save
write_dict(params_pade, "parameters_pade_constrained")
print("\nSaved as 'parameters_pade_constrained'")

# Also save the Padé coefficients
np.save("pade_coefficients.npy", final_coeffs)
print("Saved Padé coefficients to 'pade_coefficients.npy'")
