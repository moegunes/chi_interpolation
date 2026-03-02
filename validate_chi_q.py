"""Validate chi_mhg against the EXACT chi (Corradini-PZ local field factor).

Error metrics:
  q-space: max|χ_mhg(q) − χ_exact(q)| / NF   and   MADE = ⟨|δχ(q)/χ_exact(q)|⟩
  r-space: max|χ_mhg(r) − χ_exact(r)| / NF   and   MADE = ⟨|δχ(r)/χ_exact(r)|⟩

Exact χ(q) is analytic via get_chiq (Lindhard + Corradini-PZ fxc).
Exact χ(r) is obtained by FT of exact χ(q) via get_chi.
χ_mhg(q) is obtained by FT of chi_mhg(r) (DST-I).
"""

import os
import sys

os.chdir("/Users/muhammedxgunes/Desktop/research/chi_interpolation")
sys.path.insert(0, "/Users/muhammedxgunes/Desktop/research/chi_interpolation/src")

import numpy as np
from chi_mhg import chi_mhg
from chi_mhg.core import _gas_params

from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import get_chi, get_chiq

# Use DST-I compatible grid
N_q = 4096
q_max = 40.0
dq = q_max / N_q
q = np.arange(1, N_q + 1) * dq  # q_n = n*dq, n=1..N

# Dual r-grid from DST
m = np.arange(1, N_q + 1)
r_grid = m * np.pi / ((N_q + 1) * dq)

# rs range
test_rs = np.round(np.arange(0.5, 10.01, 0.25), 2).tolist()

print(
    f"{'rs':>6s} | {'max|dchiq|/NF':>14s} | {'MADE_chiq':>10s} | {'max|dchir|/NF':>14s} | {'MADE_chir':>10s}"
)
print("-" * 80)

all_max_q = []
all_made_q = []

for rs in test_rs:
    kF, n0, NF = _gas_params(rs)

    # --- Exact chi (analytic in q-space) ---
    chi_exact_q = get_chiq(q, rs)

    # --- Exact chi in r-space (FT of exact q-space) ---
    chi_exact_r = get_chi(q, rs)

    # --- chi_mhg in r-space ---
    chi_interp_r = chi_mhg(r_grid, rs)

    # --- chi_mhg in q-space (FT from r) ---
    _, chi_interp_q = chi_q_from_chi_r_fast(r_grid, chi_interp_r)

    # --- Error metrics in q-space ---
    q_mask = q < 4.0 * kF
    diff_q = np.abs(chi_interp_q - chi_exact_q)
    max_dchiq_NF = np.max(diff_q[q_mask]) / NF

    # MADE: mean |δχ(q) / χ_exact(q)|
    ref_abs_q = np.abs(chi_exact_q[q_mask])
    valid_q = ref_abs_q > 1e-6 * NF
    if np.any(valid_q):
        made_q = np.mean(diff_q[q_mask][valid_q] / ref_abs_q[valid_q]) * 100
    else:
        made_q = float("nan")

    # --- Error metrics in r-space ---
    r_mask = kF * r_grid < 15
    diff_r = np.abs(chi_interp_r - chi_exact_r)
    max_dchir_NF = np.max(diff_r[r_mask]) / NF
    ref_abs_r = np.abs(chi_exact_r[r_mask])
    valid_r = ref_abs_r > 1e-6 * NF
    if np.any(valid_r):
        made_r = np.mean(diff_r[r_mask][valid_r] / ref_abs_r[valid_r]) * 100
    else:
        made_r = float("nan")

    all_max_q.append(max_dchiq_NF)
    all_made_q.append(made_q)

    print(
        f"{rs:6.2f} | {max_dchiq_NF:14.6f} | {made_q:9.4f}% | {max_dchir_NF:14.6f} | {made_r:9.4f}%"
    )

print("-" * 80)
print(f"{'WORST':>6s} | {max(all_max_q):14.6f} | {max(all_made_q):9.4f}% |")
print(f"{'MEAN':>6s} | {np.mean(all_max_q):14.6f} | {np.mean(all_made_q):9.4f}% |")
