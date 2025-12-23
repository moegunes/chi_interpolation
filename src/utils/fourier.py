import numpy as np
from scipy.fft import dst


def chi_q_from_chi_r_fast(rlist, chi_r, qlist=None):
    """
    Inverse spherical (j0) transform matching chi_r_from_chi_q_fast.

    Given χ(r_m) on the dual DST grid, return χ(q_n) on the primal grid.

    Forward map:
       χ(r_m) = (Δq / (4π^2 r_m)) * y_m,
       y_m    = DST-I[g_n],  g_n = q_n * χ(q_n)

    Inverse steps:
       y_m = χ(r_m) * (4π^2 r_m / Δq)
       g_n = DST-I^{-1}(y_m)
       χ(q_n) = g_n / q_n
    """
    r = np.asarray(rlist, float)
    chir = np.asarray(chi_r, float)

    if r.ndim != 1 or chir.ndim != 1 or r.size != chir.size:
        raise ValueError("rlist and chi_r must be 1D arrays of same length.")

    N = r.size

    # If qlist not given, deduce Δq from r-grid duality:
    # r_m = m π / ((N+1) dq)  → dq = m π / ((N+1) r_m)
    if qlist is None:
        m = np.arange(1, N + 1)
        dq = m[0] * np.pi / ((N + 1) * r[0])  # use m=1
        qlist = dq * m
    else:
        q = np.asarray(qlist)
        dq = q[1] - q[0]

    # Step 1: reconstruct y_m
    #   χ(r_m) = (Δq / (4π^2 r_m)) * y_m   → y_m = χ(r_m) * (4π^2 r_m / Δq)
    y = chir * (4.0 * np.pi**2 * r / dq)

    # Step 2: inverse DST-I to recover g_n = q_n * χ(q_n)
    # SciPy: type=1 with norm="forward" gives the inverse scaling 1/(2*(N+1))
    # g = dst(y, type=1, norm="forward")  # shape N
    g = dst(y, type=1, norm=None) / (2.0 * (N + 1))

    # Step 3: χ(q_n) = g_n / q_n
    q = np.asarray(qlist)
    chiq = g / q

    return q, chiq


def chi_r_from_chi_q_fast(qlist, chi_q, rlist=None):
    """
    Fast O(N log N) spherical (j0) transform using a DST-I.
    Assumes q is uniformly spaced and starts at 0 or Δq.

    Mathematics:
      χ(r) = (1 / 2π^2 r) ∫_0^∞ dq [ q χ(q) ] sin(q r)
    Discretize on q_n = n Δq, n = 1..N (skip n=0), and use DST-I:
      y_m = 2 * Σ_{n=1}^N g_n sin(π n m / (N+1)),  m=1..N
      with g_n = χ(q_n) q_n
      ⇒ S_m = (Δq/2) * y_m ≈ ∫ g(q) sin(q r_m) dq
      r_m = m π / ((N+1) Δq)
      ⇒ χ(r_m) ≈ [Δq / (4 π^2 r_m)] * y_m
    """
    q = np.asarray(qlist, float)
    chiq = np.asarray(chi_q)

    # --- checks & prep
    if q.ndim != 1 or chiq.ndim != 1 or q.size != chiq.size:
        raise ValueError("qlist and chi_q must be 1D arrays of the same length.")

    dq = q[1] - q[0]

    # We need samples on q_n = n*dq, n=1..N. If the first point is 0, drop it.
    if np.isclose(q[0], 0.0):
        q_work = q[1:]
        chiq_work = chiq[1:]
    else:
        # If q[0] != dq, shift is wrong for DST-I. Require q[0] ≈ dq.
        if not np.isclose(q[0], dq):
            raise ValueError(
                "For the fast method, q must be q_n = n*Δq. "
                "Either start qlist at 0 (will drop it) or at Δq."
            )
        q_work = q
        chiq_work = chiq

    N = q_work.size
    # Build g_n = q_n * chi(q_n)
    g = q_work * chiq_work

    # Unnormalized DST-I over n=1..N gives y_m for m=1..N
    # scipy.fft.dst with type=1 returns:
    #   y[m-1] = 2 * sum_{n=1..N} g[n-1] * sin(pi*n*m/(N+1))
    y = dst(g, type=1, axis=0, norm=None)

    # Dual r-grid and χ(r_m)
    m = np.arange(1, N + 1)
    r_grid = m * np.pi / ((N + 1) * dq)
    chi_r_on_dual = (dq / (4.0 * np.pi**2)) * (y / r_grid)

    if rlist is None:
        return r_grid, chi_r_on_dual

    # Interpolate to requested rlist (monotone rlist is assumed)
    rlist = np.asarray(rlist, float)
    if np.any(rlist < r_grid[0]) or np.any(rlist > r_grid[-1]):
        # Extrapolation would be unsafe; warn via exception so users can enlarge q_max.
        raise ValueError(
            "Requested rlist extends beyond the dual grid coverage. "
            "Increase q_max (extend qlist) or restrict rlist to "
            f"[{r_grid[0]:.6g}, {r_grid[-1]:.6g}]."
        )
    chi_r_interp = np.interp(rlist, r_grid, chi_r_on_dual)
    return chi_r_interp


def r_grid_from_q(q):
    """
    Compute the dual r-grid corresponding to a uniform q-grid
    used in the DST-I spherical transform.

    Parameters
    ----------
    q : array_like
        Uniform q-grid starting at 0 or dq.

    Returns
    -------
    r_grid : ndarray
        Dual r-grid corresponding to the DST-I transform.
    """
    q = np.asarray(q, float)

    if q.ndim != 1 or q.size < 2:
        raise ValueError("q must be a 1D array with at least two points.")

    dq = q[1] - q[0]

    # Drop q=0 if present (DST-I convention)
    if np.isclose(q[0], 0.0):
        N = q.size - 1
    else:
        if not np.isclose(q[0], dq):
            raise ValueError("q must start at 0 or at dq.")
        N = q.size

    m = np.arange(1, N + 1)
    r_grid = m * np.pi / ((N + 1) * dq)
    return r_grid
