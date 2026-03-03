"""
Extend parameters_lambda10 to small rs by fitting the 2-mode model
to the full chi (with fxc via Corradini-PZ).

Also contains chi_rpa_r for future RPA-anchor work.
"""

import pickle

import numpy as np
from scipy.optimize import least_squares

from input import q, r
from optimization.fitting import BOUNDS_LOWER, BOUNDS_UPPER, _canonicalize_params
from optimization.models import delta_chi
from utils.fourier import chi_r_from_chi_q_fast
from utils.utils_chi import chi00q, corradini_pz, get_chi02, get_gas_params

# ──────────────────────────────────────────────────────────
# 1. Compute chi(r) — full (with fxc) and RPA (without fxc)
# ──────────────────────────────────────────────────────────


def chi_full_r(q_grid, rs):
    """chi_full(r) = FT[ chi0(q) / (1 - chi0(q)*(vc(q)+fxc(q))) ]"""
    chi0q = chi00q(q_grid, rs)
    vc = 4 * np.pi / q_grid**2
    fxc = corradini_pz(rs, q_grid)
    chiq = chi0q / (1 - chi0q * (vc + fxc))
    _, chir = chi_r_from_chi_q_fast(q_grid, chiq)
    return chir


def chi_rpa_r(q_grid, rs):
    """chi_RPA(r) = FT[ chi0(q) / (1 - chi0(q)*vc(q)) ]"""
    chi0q = chi00q(q_grid, rs)
    vc = 4 * np.pi / q_grid**2
    chiq_rpa = chi0q / (1 - chi0q * vc)
    _, chir_rpa = chi_r_from_chi_q_fast(q_grid, chiq_rpa)
    return chir_rpa


def delta_chi_target(q_grid, rs, use_rpa=False):
    """
    The target that the 2-mode model should reproduce:
        delta_chi = (chi - chi0^{(2)}) / (-6*pi*n0*NF)
    """
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF

    if use_rpa:
        chi_r = chi_rpa_r(q_grid, rs)
    else:
        chi_r = chi_full_r(q_grid, rs)
    chi0_r = get_chi02(q_grid, rs)

    target = (chi_r - chi0_r) / factor
    return target


# ──────────────────────────────────────────────────────────
# 2. Fit the 2-mode model
# ──────────────────────────────────────────────────────────
def fit_2mode(rs, q_grid, r_grid, initial_guesses=None, use_rpa=False):
    """
    Multi-start fit of [alpha0, f0, phi0, alpha1, f1, phi1].
    B0, B1 determined by moment constraints.
    """
    target_full = delta_chi_target(q_grid, rs, use_rpa=use_rpa)
    kF, n0, NF = get_gas_params(rs)

    # Fitting region: kF*r in [0, 6], subsampled to ~500 points
    kfr_max = 4.0
    i0 = np.argmin(np.abs(kF * r_grid))
    i1 = np.argmin(np.abs(kF * r_grid - kfr_max))
    r_region = r_grid[i0:i1]
    target_region = target_full[i0:i1]
    n_pts = len(r_region)
    step = max(1, n_pts // 200)
    r_sub = r_region[::step]
    target_sub = target_region[::step]

    if initial_guesses is None:
        initial_guesses = [np.array([1.2, 0.04, 2.0, 0.85, 0.22, 0.2])]

    def residuals(params):
        try:
            model_vals = delta_chi(r_sub, rs=rs, params=params)
            return model_vals - target_sub
        except (np.linalg.LinAlgError, ValueError):
            return np.full_like(target_sub, 1e10)

    best_result = None
    for guess in initial_guesses:
        try:
            result = least_squares(
                residuals,
                guess,
                bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
                method="trf",
                max_nfev=50000,
                ftol=1e-12,
                xtol=1e-12,
            )
            if best_result is None or result.cost < best_result.cost:
                best_result = result
        except Exception:
            continue

    return best_result


# ──────────────────────────────────────────────────────────
# 3. Main: extend parameters_lambda10 to small rs (full chi)
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.io import load_dict

    # Load existing parameters
    params_dict = load_dict("parameters")
    existing_rsl = sorted(
        k for k in params_dict if isinstance(k, (float, int)) and k != "model"
    )
    print(f"Loaded existing parameters at {len(existing_rsl)} rs values")
    print(f"  smallest existing rs: {existing_rsl[0]:.3f}")
    print(f"  largest existing rs:  {existing_rsl[-1]:.3f}")

    # New rs values below the smallest existing one, sweeping large→small
    smallest_existing = existing_rsl[0]
    new_rs = [
        rs
        for rs in [
            0.19,
            0.18,
            0.17,
            0.16,
            0.15,
            0.14,
            0.13,
            0.12,
            0.11,
            0.10,
            0.09,
            0.08,
            0.07,
            0.06,
            0.05,
        ]
        if rs < smallest_existing
    ]

    labels = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]

    print(f"\nFitting 2-mode ansatz to full chi(r) at {len(new_rs)} small rs values")
    print("(sweeping large→small rs, using warm-start from existing data)")
    print("=" * 105)
    print(
        f"{'rs':>6s}  {'alpha0':>8s}  {'f0':>8s}  {'phi0':>8s}  "
        f"{'alpha1':>8s}  {'f1':>8s}  {'phi1':>8s}  {'cost':>10s}  "
        f"{'B0':>10s}  {'B1':>10s}  {'nfev':>5s}"
    )
    print("-" * 105)

    new_params = {}
    prev_result = np.array(params_dict[existing_rsl[0]], dtype=float)

    for rs in new_rs:
        # Build initial guesses
        guesses = [prev_result.copy()]

        # Also try the closest existing params
        closest = min(existing_rsl, key=lambda x: abs(x - rs))
        guesses.append(np.array(params_dict[closest], dtype=float))

        # Clip to bounds
        for i in range(len(guesses)):
            guesses[i] = np.clip(guesses[i], BOUNDS_LOWER, BOUNDS_UPPER)

        result = fit_2mode(rs, q, r, initial_guesses=guesses, use_rpa=False)
        p = _canonicalize_params(result.x, rs)

        # Compute B0, B1
        try:
            B0, B1 = delta_chi(np.array([0.0]), rs=rs, params=p, get_constraints=True)
        except Exception:
            B0, B1 = float("nan"), float("nan")

        new_params[rs] = p
        prev_result = p.copy()

        print(
            f"{rs:6.3f}  {p[0]:8.4f}  {p[1]:8.4f}  {p[2]:8.4f}  "
            f"{p[3]:8.4f}  {p[4]:8.4f}  {p[5]:8.4f}  {result.cost:10.2e}  "
            f"{B0:10.4f}  {B1:10.4f}  {result.nfev:5d}"
        )

    # ──────────────────────────────────────────────────────────
    # 4. Convergence check
    # ──────────────────────────────────────────────────────────
    all_rs = sorted(set(existing_rsl) | set(new_params.keys()))
    all_p = {}
    for rs in all_rs:
        if rs in new_params:
            all_p[rs] = new_params[rs]
        else:
            all_p[rs] = np.array(params_dict[rs], dtype=float)

    print("\n\n=== Parameter continuity at the junction ===")
    # Show a few existing + all new, sorted
    show_rs = [rs for rs in existing_rsl if rs <= smallest_existing + 0.1][:3] + sorted(
        new_params.keys()
    )
    print(
        f"{'rs':>6s}  {'src':>4s}  {'alpha0':>8s}  {'f0':>8s}  {'phi0':>8s}  "
        f"{'alpha1':>8s}  {'f1':>8s}  {'phi1':>8s}"
    )
    print("-" * 75)
    for rs in sorted(show_rs, reverse=True):
        src = "old" if rs in params_dict else "NEW"
        p = all_p[rs]
        print(
            f"{rs:6.3f}  {src:>4s}  {p[0]:8.4f}  {p[1]:8.4f}  {p[2]:8.4f}  "
            f"{p[3]:8.4f}  {p[4]:8.4f}  {p[5]:8.4f}"
        )

    # ──────────────────────────────────────────────────────────
    # 5. Save combined dictionary as .pkl
    # ──────────────────────────────────────────────────────────
    combined = dict(params_dict)  # copy existing (includes 'model' key etc.)
    for rs, p in new_params.items():
        combined[rs] = p.tolist()

    outfile = "parameters_extended.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(combined, f)
    print(f"\nSaved combined dictionary ({len(all_rs)} rs values) to {outfile}")
