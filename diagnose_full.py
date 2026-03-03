"""Diagnose all 8 quantities: 6 nonlinear params + B0, B1 + condition number."""

import sys

import numpy as np

sys.path.insert(0, "src")
from utils.io import load_dict
from utils.physics import J_n_m_kFr, delta_C

params = load_dict("parameters")
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])

print(f"rs range: {rsl[0]:.2f} to {rsl[-1]:.2f}, {len(rsl)} points\n")

header = f"{'rs':>5s}  {'alpha0':>8s} {'f0':>8s} {'phi0':>8s} {'alpha1':>8s} {'f1':>8s} {'phi1':>8s} | {'B0':>10s} {'B1':>10s} {'cond(M)':>10s}"
print(header)
print("-" * len(header))

B0_all, B1_all = [], []
cond_all = []
data_all = []

for rs in rsl:
    p = params[rs]
    alpha0, f0, phi0, alpha1, f1, phi1 = p
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    k0, k1 = 2 * np.pi * f0, 2 * np.pi * f1

    J0 = J_n_m_kFr(0, k0, alpha0, phi0, kF)
    J1 = J_n_m_kFr(0, k1, alpha1, phi1, kF)
    J3 = J_n_m_kFr(1, k0, alpha0, phi0, kF)
    J4 = J_n_m_kFr(1, k1, alpha1, phi1, kF)
    Mmat = np.array([[J3, J4], [J0, J1]])
    b = np.array([delta_C(1, rs), delta_C(0, rs)])
    B0, B1 = np.linalg.solve(Mmat, b)
    cn = np.linalg.cond(Mmat)

    B0_all.append(B0)
    B1_all.append(B1)
    cond_all.append(cn)
    data_all.append(p)

    print(
        f"{rs:5.2f}  {alpha0:8.4f} {f0:8.4f} {phi0:8.4f} "
        f"{alpha1:8.4f} {f1:8.4f} {phi1:8.4f} | "
        f"{B0:10.4f} {B1:10.4f} {cn:10.1f}"
    )

B0_all = np.array(B0_all)
B1_all = np.array(B1_all)
data_all = np.array(data_all)

print("\n=== Max jumps between consecutive rs values ===")
labels = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1", "B0", "B1"]
all_arrays = [data_all[:, i] for i in range(6)] + [B0_all, B1_all]

for name, arr in zip(labels, all_arrays):
    diffs = np.abs(np.diff(arr))
    idx = np.argmax(diffs)
    rng = np.max(arr) - np.min(arr)
    pct = diffs[idx] / (rng + 1e-30) * 100
    print(
        f"  {name:8s}: max jump = {diffs[idx]:.6f} "
        f"({pct:.1f}% of range) between rs={rsl[idx]:.2f} and rs={rsl[idx + 1]:.2f}"
    )
