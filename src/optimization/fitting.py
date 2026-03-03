import numpy as np
from scipy.optimize import curve_fit, minimize

from optimization.models import delta_pi
from utils.utils_chi import get_chi02, get_gas_params, get_pi

# --- Parameter bounds ---
# params = [alpha0, f0, phi0, alpha1, f1, phi1]
# alpha > 0 (damping), f >= F_MIN (prevents mode collapse), phi in [-pi, pi]
F_MIN = 0.02  # Minimum frequency — prevents degenerate pure-exponential modes
DF_MIN = 0.03  # Minimum |f0 - f1| — prevents both modes collapsing to same frequency
BOUNDS_LOWER = np.array([1e-4, F_MIN, -np.pi, 1e-4, F_MIN, -np.pi])
BOUNDS_UPPER = np.array([20.0, 3.0, np.pi, 20.0, 3.0, np.pi])

kfR_max = 6.0


def _canonicalize_params(params, rs=None):
    """Enforce canonical form: f >= 0, phases in [-pi, pi], B >= 0, modes ordered by |f|."""
    p = np.array(params, dtype=float)
    # 1. Sign symmetry: cos(f*r + phi) = cos(-f*r - phi), so make f >= 0
    for mode_idx, (f_idx, phi_idx) in enumerate([(1, 2), (4, 5)]):
        if p[f_idx] < 0:
            p[f_idx] = -p[f_idx]
            p[phi_idx] = -p[phi_idx]
        # Enforce minimum frequency
        p[f_idx] = max(p[f_idx], F_MIN)
    # 2. Canonicalize phases to [-pi, pi)
    p[2] = (p[2] + np.pi) % (2 * np.pi) - np.pi
    p[5] = (p[5] + np.pi) % (2 * np.pi) - np.pi
    # 3. B >= 0 canonicalization: exploit (phi, B) -> (phi+pi, -B) symmetry
    #    Shifting phi_i by pi negates B_i without changing the model output.
    if rs is not None:
        B = _compute_B(p, rs)
        if B is not None:
            if B[0] < 0:
                p[2] += np.pi
            if B[1] < 0:
                p[5] += np.pi
            # Re-canonicalize phases to [-pi, pi)
            p[2] = (p[2] + np.pi) % (2 * np.pi) - np.pi
            p[5] = (p[5] + np.pi) % (2 * np.pi) - np.pi
    # 4. Canonical mode ordering: mode 0 has smaller |f|
    if abs(p[1]) > abs(p[4]):
        p = np.array([p[3], p[4], p[5], p[0], p[1], p[2]])
    return p


def _compute_B(params, rs):
    """Safely compute (B0, B1) from the 6 nonlinear params at a given rs."""
    try:
        B0, B1 = delta_pi(np.array([0.0]), rs=rs, params=params, get_constraints=True)
        if np.isfinite(B0) and np.isfinite(B1):
            return (float(B0), float(B1))
    except (np.linalg.LinAlgError, ValueError):
        pass
    return None


def _physics_initial_guess(rs):
    """Physics-motivated initial guess that is reasonable for any rs."""
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    return np.array(
        [
            1.5 / kF,
            3 / 2.0 / np.pi,
            np.pi / 2 - 0.1,
            1 / kF,
            -1 / 2.0 / np.pi,
            np.pi / 2 + 0.1,
        ]
    )


def _fit_one_point(rs, q, r, model, prev_params, cost_tolerance, prev_B=None):
    """Fit a single rs point with warm start from prev_params.

    Returns (best_params, best_cov).
    """
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    piR = get_pi(q, rs)
    chi0R = get_chi02(q, rs)
    delta_pi_exact = -(chi0R - piR) / factor

    physics_guess = _physics_initial_guess(rs)
    candidates = [prev_params, physics_guess]

    # Random perturbation around warm start
    scale = np.array([0.1, 0.05, 0.2, 0.1, 0.05, 0.2])
    candidates.append(prev_params + scale * np.random.randn(6))

    # Fitting region: subsample for efficiency
    fit_idx0 = np.argmin(np.abs(kF * r - 0))
    fit_idx1 = np.argmin(np.abs(kF * r - kfR_max))
    r_fit_full = r[fit_idx0:fit_idx1]
    y_fit_full = delta_pi_exact[fit_idx0:fit_idx1]
    n_pts = len(r_fit_full)
    step = max(1, n_pts // 2000)
    r_fit = r_fit_full[::step]
    y_fit = y_fit_full[::step]

    results = []

    # --- Standard curve_fit candidates ---
    for p0 in candidates:
        try:
            p_opt, p_cov = guess_X(
                r, rs, delta_pi_exact, model, p0, kFr0=0, kFr1=kfR_max
            )
            p_opt = _canonicalize_params(p_opt, rs)
            residual = model(r_fit, rs=rs, params=p_opt) - y_fit
            data_cost = np.sum(residual**2)
            results.append((p_opt, data_cost, p_cov))
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            pass

    # --- Regularized minimize from warm start (multiple strengths) ---
    for lam in [0.1, 1.0, 10.0]:
        try:
            p_reg = _regularized_fit(
                r_fit, y_fit, rs, model, prev_params, lambda_smooth=lam
            )
            p_reg = _canonicalize_params(p_reg, rs)
            residual = model(r_fit, rs=rs, params=p_reg) - y_fit
            data_cost = np.sum(residual**2)
            results.append((p_reg, data_cost, np.zeros((6, 6))))
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            pass

    if not results:
        print(f"Warning: all restarts failed at rs={rs:.2f}, using physics guess")
        return _canonicalize_params(physics_guess, rs), np.zeros((6, 6))

    return _select_smoothest(results, prev_params, cost_tolerance, rs=rs, prev_B=prev_B)


def _global_smooth_refit(
    rslist_sorted, q, r, model, parameters, lambda_smooth=10.0, max_iter=500
):
    """Phase 6: Global simultaneous re-fit with smoothness penalty.

    Optimizes all rs points simultaneously via L-BFGS-B:
        L = sum_i DataCost(rs_i)
          + lambda_smooth * SmoothnessNorm * sum_i ||S*(theta_{i+1} - theta_i)||^2 / drs_i^2

    where S normalizes each parameter by its standard deviation.

    lambda_smooth: relative weight of smoothness vs data fidelity (1.0 = equal).
    """
    from utils.physics import delta_C as _delta_C

    n_rs = len(rslist_sorted)
    n_p = 6

    # --- Precompute target data and model constants for all rs ---
    print("Phase 6: Precomputing target data for global smooth re-fit...")
    fit_data = []  # (y_fit,) — target values
    kFr_arrays = []  # kF*r arrays for model evaluation
    delta_Cs = []  # (dC1, dC0) moment constraints
    inv_kF3 = []  # 1/kF^3
    inv_kF5 = []  # 1/kF^5

    for rs_val in rslist_sorted:
        kF, n0, NF = get_gas_params(rs_val)
        factor = -6 * np.pi * n0 * NF
        piR = get_pi(q, rs_val)
        chi0R = get_chi02(q, rs_val)
        dpi = -(chi0R - piR) / factor
        i0 = np.argmin(np.abs(kF * r - 0))
        i1 = np.argmin(np.abs(kF * r - kfR_max))
        rf = r[i0:i1]
        yf = dpi[i0:i1]
        step = max(1, len(rf) // 300)
        fit_data.append(yf[::step])
        kFr_arrays.append(kF * rf[::step])
        delta_Cs.append((_delta_C(1, rs_val), _delta_C(0, rs_val)))
        inv_kF3.append(1.0 / kF**3)
        inv_kF5.append(1.0 / kF**5)

    # --- Initial parameter vector (flattened) ---
    x0 = np.concatenate([parameters[rs] for rs in rslist_sorted])
    lb = np.tile(BOUNDS_LOWER, n_rs)
    ub = np.tile(BOUNDS_UPPER, n_rs)

    # --- Per-parameter scaling ---
    pm = x0.reshape(n_rs, n_p)
    scales = np.std(pm, axis=0)
    scales = np.where(scales > 1e-10, scales, 1.0)
    inv_s2 = 1.0 / scales**2

    # --- Rs spacing ---
    drs = np.diff(rslist_sorted)
    inv_drs2 = 1.0 / drs**2

    # --- Lambda calibration: normalize so lambda_smooth=1 gives equal weight ---
    # Compute initial data cost
    init_data = 0.0
    for i in range(n_rs):
        yf = fit_data[i]
        kfr = kFr_arrays[i]
        p = pm[i]
        k0, k1 = 2 * np.pi * p[1], 2 * np.pi * p[4]
        e0 = np.exp(1j * p[2])
        e1 = np.exp(1j * p[5])
        z0 = p[0] - 1j * k0
        z1 = p[3] - 1j * k1
        J0 = 2.0 * np.real(e0 / z0**3) * inv_kF3[i]
        J1 = 2.0 * np.real(e1 / z1**3) * inv_kF3[i]
        J3 = 24.0 * np.real(e0 / z0**5) * inv_kF5[i]
        J4 = 24.0 * np.real(e1 / z1**5) * inv_kF5[i]
        det = J3 * J1 - J4 * J0
        dc1, dc0 = delta_Cs[i]
        B0 = (dc1 * J1 - dc0 * J4) / det
        B1 = (dc0 * J3 - dc1 * J0) / det
        pred = B0 * np.exp(-p[0] * kfr) * np.cos(k0 * kfr + p[2]) + B1 * np.exp(
            -p[3] * kfr
        ) * np.cos(k1 * kfr + p[5])
        init_data += np.sum((pred - yf) ** 2)

    # Compute initial smoothness cost (unweighted)
    init_smooth = 0.0
    for i in range(n_rs - 1):
        d = pm[i + 1] - pm[i]
        init_smooth += np.sum(d**2 * inv_s2) * inv_drs2[i]

    # Normalize: at lambda_smooth=1, both terms are equal at the starting point
    lam = lambda_smooth * init_data / (init_smooth + 1e-30)
    lam_degen = 50.0 * lam  # extra weight for degenerate mode-0 params

    # Precompute curvature penalty spacing
    avg_drs_sq = np.zeros(n_rs - 2)
    for i in range(n_rs - 2):
        avg_drs_sq[i] = (0.5 * (drs[i] + drs[i + 1])) ** 2
    inv_avg_drs4 = 1.0 / avg_drs_sq**2

    print(f"  Data: {init_data:.4e}, Smooth: {init_smooth:.4e}, λ_eff: {lam:.4e}")
    print(f"  Optimizing {n_rs * n_p} variables...")

    # --- Fast inline objective ---
    def objective(x):
        P = x.reshape(n_rs, n_p)
        total = 0.0

        # Data cost (inlined model: avoid function call overhead)
        for i in range(n_rs):
            a0, f0, ph0, a1, f1, ph1 = P[i]
            k0 = 2.0 * np.pi * f0
            k1 = 2.0 * np.pi * f1
            e0 = np.exp(1j * ph0)
            e1 = np.exp(1j * ph1)
            z0 = a0 - 1j * k0
            z1 = a1 - 1j * k1
            J0v = 2.0 * np.real(e0 / z0**3) * inv_kF3[i]
            J1v = 2.0 * np.real(e1 / z1**3) * inv_kF3[i]
            J3v = 24.0 * np.real(e0 / z0**5) * inv_kF5[i]
            J4v = 24.0 * np.real(e1 / z1**5) * inv_kF5[i]
            det = J3v * J1v - J4v * J0v
            if abs(det) < 1e-30:
                return 1e30
            dc1, dc0 = delta_Cs[i]
            B0 = (dc1 * J1v - dc0 * J4v) / det
            B1 = (dc0 * J3v - dc1 * J0v) / det
            kfr = kFr_arrays[i]
            pred = B0 * np.exp(-a0 * kfr) * np.cos(k0 * kfr + ph0) + B1 * np.exp(
                -a1 * kfr
            ) * np.cos(k1 * kfr + ph1)
            total += np.sum((pred - fit_data[i]) ** 2)

        # Smoothness penalty (scaled, per-unit-rs)
        for i in range(n_rs - 1):
            d = P[i + 1] - P[i]
            total += lam * np.sum(d**2 * inv_s2) * inv_drs2[i]

            # Enhanced penalty for degenerate mode-0 (f0 near F_MIN)
            f0_next = P[i + 1, 1]
            w = np.exp(-(((f0_next - F_MIN) / 0.01) ** 2))
            if w > 0.01:
                total += (
                    lam_degen
                    * w
                    * (d[0] ** 2 * inv_s2[0] + d[2] ** 2 * inv_s2[2])
                    * inv_drs2[i]
                )

        # Mode separation penalty (smooth)
        for i in range(n_rs):
            df = np.sqrt((P[i, 1] - P[i, 4]) ** 2 + 1e-12)
            if df < DF_MIN:
                total += 1e4 * (DF_MIN - df) ** 2

        # Second-order (curvature) penalty — targets staircase patterns
        for i in range(n_rs - 2):
            d2 = P[i + 2] - 2 * P[i + 1] + P[i]
            total += lam * np.sum(d2**2 * inv_s2) * inv_avg_drs4[i]

        return total

    # --- Optimize ---
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=list(zip(lb, ub)),
        options={"maxiter": max_iter, "ftol": 1e-15, "maxfun": 500000},
    )

    # --- Report ---
    P_opt = result.x.reshape(n_rs, n_p)
    final_data = 0.0
    for i in range(n_rs):
        yf = fit_data[i]
        kfr = kFr_arrays[i]
        p = P_opt[i]
        k0, k1 = 2 * np.pi * p[1], 2 * np.pi * p[4]
        e0 = np.exp(1j * p[2])
        e1 = np.exp(1j * p[5])
        z0 = p[0] - 1j * k0
        z1 = p[3] - 1j * k1
        J0v = 2.0 * np.real(e0 / z0**3) * inv_kF3[i]
        J1v = 2.0 * np.real(e1 / z1**3) * inv_kF3[i]
        J3v = 24.0 * np.real(e0 / z0**5) * inv_kF5[i]
        J4v = 24.0 * np.real(e1 / z1**5) * inv_kF5[i]
        det = J3v * J1v - J4v * J0v
        dc1, dc0 = delta_Cs[i]
        B0 = (dc1 * J1v - dc0 * J4v) / det
        B1 = (dc0 * J3v - dc1 * J0v) / det
        pred = B0 * np.exp(-p[0] * kfr) * np.cos(k0 * kfr + p[2]) + B1 * np.exp(
            -p[3] * kfr
        ) * np.cos(k1 * kfr + p[5])
        final_data += np.sum((pred - yf) ** 2)

    final_smooth = 0.0
    for i in range(n_rs - 1):
        d = P_opt[i + 1] - P_opt[i]
        final_smooth += np.sum(d**2 * inv_s2) * inv_drs2[i]

    print(f"  {result.message}")
    print(
        f"  Data: {init_data:.4e} → {final_data:.4e}"
        f" ({(final_data / init_data - 1) * 100:+.1f}%)"
    )
    print(
        f"  Smooth: {init_smooth:.4e} → {final_smooth:.4e}"
        f" ({(final_smooth / init_smooth - 1) * 100:+.1f}%)"
    )

    # --- Canonicalize and store ---
    for i, rs in enumerate(rslist_sorted):
        parameters[rs] = _canonicalize_params(P_opt[i], rs)

    return parameters


def fit_params(
    rslist, q, r, model=delta_pi, inverse=False, n_restarts=3, cost_tolerance=3.0
):
    """Fit parameters using bidirectional sweep from a reliable anchor point.

    Strategy:
    1. Pick an anchor rs near 2.0 (physics guess works reliably there)
    2. Multi-start fit at anchor with extra candidates for robustness
    3. Sweep upward: anchor → max rs (warm start propagates up)
    4. Sweep downward: anchor → min rs (warm start propagates down)

    cost_tolerance: a candidate is acceptable if its data cost is within
                    this factor of the best data cost found. Among acceptable
                    candidates, the one closest to the previous rs params is chosen.
    """
    from tqdm import tqdm

    parameters = {}
    parameters_cov = {}
    parameters["model"] = model

    rslist_sorted = np.sort(rslist)

    # --- Phase 1: Fit anchor point with aggressive multi-start ---
    anchor_target = 2.0
    anchor_idx = int(np.argmin(np.abs(rslist_sorted - anchor_target)))
    anchor_rs = rslist_sorted[anchor_idx]

    rs = anchor_rs
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    piR = get_pi(q, rs)
    chi0R = get_chi02(q, rs)
    delta_pi_exact = -(chi0R - piR) / factor

    physics_guess = _physics_initial_guess(rs)
    candidates = [physics_guess]
    # Extra perturbations for robust anchor
    scale = np.array([0.3, 0.7, 0.5, 0.3, 0.7, 0.5])
    for _ in range(10):
        candidates.append(physics_guess + scale * np.random.randn(6))

    fit_idx0 = np.argmin(np.abs(kF * r - 0))
    fit_idx1 = np.argmin(np.abs(kF * r - kfR_max))
    r_fit_full = r[fit_idx0:fit_idx1]
    y_fit_full = delta_pi_exact[fit_idx0:fit_idx1]
    n_pts = len(r_fit_full)
    step = max(1, n_pts // 2000)
    r_fit = r_fit_full[::step]
    y_fit = y_fit_full[::step]

    results = []
    for p0 in candidates:
        try:
            p_opt, p_cov = guess_X(
                r, rs, delta_pi_exact, model, p0, kFr0=0, kFr1=kfR_max
            )
            p_opt = _canonicalize_params(p_opt, rs)
            residual = model(r_fit, rs=rs, params=p_opt) - y_fit
            data_cost = np.sum(residual**2)
            results.append((p_opt, data_cost, p_cov))
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            pass

    if not results:
        raise RuntimeError(f"All restarts failed at anchor rs={anchor_rs:.2f}")

    best = min(results, key=lambda x: x[1])
    parameters[anchor_rs] = best[0]
    parameters_cov[anchor_rs] = best[2]
    B_values = {}
    B_values[anchor_rs] = _compute_B(best[0], anchor_rs)
    print(f"Anchor at rs={anchor_rs:.2f}: {best[0]}")

    # --- Phase 2: Sweep upward from anchor → max rs ---
    up_range = range(anchor_idx + 1, len(rslist_sorted))
    for i in tqdm(up_range, desc="Up sweep  ", ncols=80):
        rs = rslist_sorted[i]
        pred_rs = rslist_sorted[i - 1]
        best_params, best_cov = _fit_one_point(
            rs,
            q,
            r,
            model,
            parameters[pred_rs],
            cost_tolerance,
            prev_B=B_values.get(pred_rs),
        )
        parameters[rs] = best_params
        parameters_cov[rs] = best_cov if best_cov is not None else np.zeros((6, 6))
        B_values[rs] = _compute_B(best_params, rs)

    # --- Phase 3: Sweep downward from anchor → min rs ---
    down_range = range(anchor_idx - 1, -1, -1)
    for i in tqdm(down_range, desc="Down sweep", ncols=80):
        rs = rslist_sorted[i]
        pred_rs = rslist_sorted[i + 1]
        best_params, best_cov = _fit_one_point(
            rs,
            q,
            r,
            model,
            parameters[pred_rs],
            cost_tolerance,
            prev_B=B_values.get(pred_rs),
        )
        parameters[rs] = best_params
        parameters_cov[rs] = best_cov if best_cov is not None else np.zeros((6, 6))
        B_values[rs] = _compute_B(best_params, rs)

    # --- Phase 4: Re-fit anchor using neighbors for consistency ---
    if anchor_idx > 0 and anchor_idx < len(rslist_sorted) - 1:
        left_params = parameters[rslist_sorted[anchor_idx - 1]]
        right_params = parameters[rslist_sorted[anchor_idx + 1]]
        avg_neighbour = 0.5 * (left_params + right_params)
        left_B = B_values.get(rslist_sorted[anchor_idx - 1])
        right_B = B_values.get(rslist_sorted[anchor_idx + 1])
        avg_B = None
        if left_B is not None and right_B is not None:
            avg_B = (0.5 * (left_B[0] + right_B[0]), 0.5 * (left_B[1] + right_B[1]))
        best_params, best_cov = _fit_one_point(
            anchor_rs, q, r, model, avg_neighbour, cost_tolerance, prev_B=avg_B
        )
        parameters[anchor_rs] = best_params
        parameters_cov[anchor_rs] = (
            best_cov if best_cov is not None else np.zeros((6, 6))
        )

    # --- Phase 5: Extrapolation-based re-fit for low-rs region ---
    # Use polynomial extrapolation from the smooth region to provide
    # physically reasonable initial guesses for low-rs points.
    smooth_threshold = 2.0  # rs above which params are well-determined
    smooth_mask = rslist_sorted >= smooth_threshold
    if np.sum(smooth_mask) >= 4:
        smooth_rs = rslist_sorted[smooth_mask]
        smooth_params = np.array([parameters[rs] for rs in smooth_rs])
        # Fit cubic polynomial to each parameter
        param_polys = []
        for j in range(6):
            coeffs = np.polyfit(smooth_rs, smooth_params[:, j], 3)
            param_polys.append(np.poly1d(coeffs))
        # Re-fit points below threshold using extrapolated params
        low_rs_indices = np.where(~smooth_mask)[0]
        # Sweep downward from threshold for continuity
        for i in reversed(low_rs_indices):
            rs = rslist_sorted[i]
            pred_rs = rslist_sorted[i + 1]
            extrap_params = np.array([poly(rs) for poly in param_polys])
            extrap_params = np.clip(
                extrap_params, BOUNDS_LOWER + 1e-6, BOUNDS_UPPER - 1e-6
            )
            extrap_params = _canonicalize_params(extrap_params, rs)
            prev_B = B_values.get(pred_rs)
            # Generate candidates: extrapolated, interpolated from neighbor, current
            kF, n0, NF = get_gas_params(rs)
            factor = -6 * np.pi * n0 * NF
            piR = get_pi(q, rs)
            chi0R = get_chi02(q, rs)
            delta_pi_exact = -(chi0R - piR) / factor
            fit_idx0 = np.argmin(np.abs(kF * r - 0))
            fit_idx1 = np.argmin(np.abs(kF * r - kfR_max))
            r_fit_full = r[fit_idx0:fit_idx1]
            y_fit_full = delta_pi_exact[fit_idx0:fit_idx1]
            n_pts = len(r_fit_full)
            step = max(1, n_pts // 2000)
            r_fit = r_fit_full[::step]
            y_fit = y_fit_full[::step]
            results = []
            # Candidate 1: current sweep result
            fwd_params = parameters[rs]
            res_fwd = model(r_fit, rs=rs, params=fwd_params) - y_fit
            results.append((fwd_params, np.sum(res_fwd**2), parameters_cov[rs]))
            # Candidate 2-3: curve_fit from extrapolated and neighbor
            for p0 in [extrap_params, parameters[pred_rs]]:
                try:
                    p_opt, p_cov = guess_X(
                        r, rs, delta_pi_exact, model, p0, kFr0=0, kFr1=kfR_max
                    )
                    p_opt = _canonicalize_params(p_opt, rs)
                    res = model(r_fit, rs=rs, params=p_opt) - y_fit
                    results.append((p_opt, np.sum(res**2), p_cov))
                except (RuntimeError, np.linalg.LinAlgError, ValueError):
                    pass
            # Candidate 4-6: regularized toward neighbor at various strengths
            for lam in [1.0, 10.0, 50.0]:
                try:
                    p_reg = _regularized_fit(
                        r_fit,
                        y_fit,
                        rs,
                        model,
                        parameters[pred_rs],
                        lambda_smooth=lam,
                    )
                    p_reg = _canonicalize_params(p_reg, rs)
                    res = model(r_fit, rs=rs, params=p_reg) - y_fit
                    results.append((p_reg, np.sum(res**2), np.zeros((6, 6))))
                except (RuntimeError, np.linalg.LinAlgError, ValueError):
                    pass
            best_params, best_cov = _select_smoothest(
                results, parameters[pred_rs], cost_tolerance * 2, rs=rs, prev_B=prev_B
            )
            parameters[rs] = best_params
            parameters_cov[rs] = best_cov if best_cov is not None else np.zeros((6, 6))
            B_values[rs] = _compute_B(best_params, rs)

    # --- Phase 6: Global smooth re-fit ---
    parameters = _global_smooth_refit(
        rslist_sorted, q, r, model, parameters, lambda_smooth=50.0, max_iter=2000
    )

    return parameters, parameters_cov


def _select_smoothest(results, prev_params, cost_tolerance=2.0, rs=None, prev_B=None):
    """Among candidates with data cost within `cost_tolerance` of the best,
    pick the one closest (in L2 norm including B0/B1) to prev_params.
    If prev_params is None, just pick the lowest data cost."""
    if prev_params is None:
        # No previous: just pick the lowest data cost
        best = min(results, key=lambda x: x[1])
        return best[0], best[2]

    # Find the best data cost
    best_data_cost = min(r[1] for r in results)
    threshold = cost_tolerance * best_data_cost + 1e-30

    # Filter acceptable candidates
    acceptable = [r for r in results if r[1] <= threshold]

    # Among acceptable, pick closest to prev_params (including B0/B1)
    def proximity(result):
        d = np.sum((result[0] - prev_params) ** 2)
        # B0/B1 proximity: penalise jumps in the linearly-solved coefficients
        if rs is not None and prev_B is not None:
            B = _compute_B(result[0], rs)
            if B is not None:
                d += (B[0] - prev_B[0]) ** 2 + (B[1] - prev_B[1]) ** 2
            else:
                d += 1e6  # singular constraint matrix
        # Mode separation penalty
        f0, f1 = result[0][1], result[0][4]
        if abs(f1 - f0) < DF_MIN:
            d += 1e4
        return d

    best = min(acceptable, key=proximity)
    return best[0], best[2]


def _backward_fixup(
    rslist, q, r, model, parameters, parameters_cov, cost_tolerance, n_edge=3
):
    """Backward pass over the first ``n_edge`` points to smooth out edge jumps.

    After the forward sweep the first point has no predecessor, so it may land
    on a different branch.  This function re-fits the first ``n_edge`` points in
    reverse order using the already-fitted successor as warm start and applies
    the same proximity-first selection logic.
    """
    end = min(n_edge, len(rslist) - 1)  # how far back from the start to fix
    for idx_rs in range(end, -1, -1):
        rs = rslist[idx_rs]
        kF, n0, NF = get_gas_params(rs)
        factor = -6 * np.pi * n0 * NF
        piR = get_pi(q, rs)
        chi0R = get_chi02(q, rs)
        delta_pi_exact = -(chi0R - piR) / factor

        # Use the successor's params as the "previous" for proximity
        if idx_rs < len(rslist) - 1:
            neighbour_params = parameters[rslist[idx_rs + 1]]
        else:
            continue  # last point has no successor, skip

        # Subsample fitting region
        fit_idx0 = np.argmin(np.abs(kF * r - 0))
        fit_idx1 = np.argmin(np.abs(kF * r - kfR_max))
        r_fit_full = r[fit_idx0:fit_idx1]
        y_fit_full = delta_pi_exact[fit_idx0:fit_idx1]
        n_pts = len(r_fit_full)
        step = max(1, n_pts // 2000)
        r_fit = r_fit_full[::step]
        y_fit = y_fit_full[::step]

        # Collect candidates
        results = []

        # Current forward-sweep solution
        fwd_params = parameters[rs]
        residual_fwd = model(r_fit, rs=rs, params=fwd_params) - y_fit
        results.append((fwd_params, np.sum(residual_fwd**2), parameters_cov[rs]))

        # Warm start from successor
        try:
            p_opt, p_cov = guess_X(
                r, rs, delta_pi_exact, model, neighbour_params, kFr0=0, kFr1=kfR_max
            )
            p_opt = _canonicalize_params(p_opt, rs)
            residual = model(r_fit, rs=rs, params=p_opt) - y_fit
            results.append((p_opt, np.sum(residual**2), p_cov))
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            pass

        # Regularized toward successor (weak and strong)
        for lam in [0.1, 1.0, 10.0]:
            try:
                p_reg = _regularized_fit(
                    r_fit, y_fit, rs, model, neighbour_params, lambda_smooth=lam
                )
                p_reg = _canonicalize_params(p_reg, rs)
                residual = model(r_fit, rs=rs, params=p_reg) - y_fit
                results.append((p_reg, np.sum(residual**2), np.zeros((6, 6))))
            except (RuntimeError, np.linalg.LinAlgError, ValueError):
                pass

        # Random perturbations around successor
        scale = np.array([0.1, 0.05, 0.2, 0.1, 0.05, 0.2])
        for _ in range(3):
            p0 = neighbour_params + scale * np.random.randn(6)
            try:
                p_opt, p_cov = guess_X(
                    r, rs, delta_pi_exact, model, p0, kFr0=0, kFr1=kfR_max
                )
                p_opt = _canonicalize_params(p_opt, rs)
                residual = model(r_fit, rs=rs, params=p_opt) - y_fit
                results.append((p_opt, np.sum(residual**2), p_cov))
            except (RuntimeError, np.linalg.LinAlgError, ValueError):
                pass

        best_params, best_cov = _select_smoothest(
            results, neighbour_params, cost_tolerance
        )
        parameters[rs] = best_params
        parameters_cov[rs] = best_cov


def _regularized_fit(r_fit, y_fit, rs, model, prev_params, lambda_smooth):
    """Run scipy.optimize.minimize with embedded L2 regularization toward prev_params."""
    y_norm = np.sum(y_fit**2) + 1e-30
    p_norm = np.sum(prev_params**2) + 1e-30
    lam = lambda_smooth * (y_norm / p_norm)

    def objective(params):
        try:
            residual = model(r_fit, rs=rs, params=params) - y_fit
            data_cost = np.sum(residual**2)
            reg_cost = lam * np.sum((params - prev_params) ** 2)
            # Mode separation penalty: prevent both modes collapsing to same frequency
            df = abs(params[1] - params[4])
            sep_penalty = lam * 10.0 * max(0, DF_MIN - df) ** 2
            # Degenerate mode penalty: when f_i is near F_MIN, alpha_i and phi_i
            # are poorly determined (mode contributes negligibly).
            # Strongly regularize those parameters to follow the smooth trend.
            degen_penalty = 0.0
            for f_idx, a_idx, p_idx in [(1, 0, 2), (4, 3, 5)]:
                if abs(params[f_idx] - F_MIN) < F_MIN / 2:
                    degen_penalty += (
                        100.0
                        * lam
                        * (
                            (params[a_idx] - prev_params[a_idx]) ** 2
                            + (params[p_idx] - prev_params[p_idx]) ** 2
                        )
                    )
            return data_cost + reg_cost + sep_penalty + degen_penalty
        except (np.linalg.LinAlgError, ValueError):
            return 1e30

    bounds_list = list(zip(BOUNDS_LOWER, BOUNDS_UPPER))
    p0 = np.clip(prev_params, BOUNDS_LOWER + 1e-6, BOUNDS_UPPER - 1e-6)
    result = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        bounds=bounds_list,
        options={"maxiter": 5000, "ftol": 1e-15},
    )
    return result.x


def guess_X(
    r, rs, X_exact, model, initial_guess, kFr0=0, kFr1=kfR_max, max_fit_pts=2000
):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    fit_idx0 = np.argmin(np.abs(kF * r - kFr0))
    fit_idx1 = np.argmin(np.abs(kF * r - kFr1))

    r_fit = r[fit_idx0:fit_idx1]
    y_fit = X_exact[fit_idx0:fit_idx1]

    # Subsample if the fitting array is too large (preserves endpoints)
    n_pts = len(r_fit)
    if n_pts > max_fit_pts:
        step = n_pts // max_fit_pts
        r_fit = r_fit[::step]
        y_fit = y_fit[::step]

    def model_wrapper(r, alpha0, f0, phi0, alpha1, f1, phi1):
        params = [alpha0, f0, phi0, alpha1, f1, phi1]
        return model(r, rs=rs, params=params)

    # Clip initial guess to within bounds
    p0 = np.clip(initial_guess, BOUNDS_LOWER + 1e-6, BOUNDS_UPPER - 1e-6)

    p_opt, p_cov = curve_fit(
        model_wrapper,
        r_fit,
        y_fit,
        p0=p0,
        bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
        method="trf",
        maxfev=30000,
    )

    return p_opt, p_cov
