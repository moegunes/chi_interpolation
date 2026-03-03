"""Verify kappa-included pi_moment formulas against numerical integration.

We compare the analytic C_n from the Mathematica output with the
numerical moments of Pi(q) computed via get_piq.
"""

import os
import sys

os.chdir("/Users/muhammedxgunes/Desktop/research/chi_interpolation")
sys.path.insert(0, "src")

import numpy as np

from utils.utils_chi import chi00q, corradini_pz

kappa = 0.0225


def get_fxc_derivatives(rs, dq=1e-3):
    """Numerically compute f0, f2, f4, f6 from corradini_pz."""
    f0 = corradini_pz(rs, 0.0)

    q = np.array([-dq, 0.0, dq])
    fvals = corradini_pz(rs, q)
    f2 = 0.5 * (fvals[2] - 2 * fvals[1] + fvals[0]) / dq**2

    q = np.array([-2 * dq, -dq, 0.0, dq, 2 * dq])
    fvals = corradini_pz(rs, q)
    f4 = (
        (fvals[0] - 4 * fvals[1] + 6 * fvals[2] - 4 * fvals[3] + fvals[4])
        / dq**4
        / 24.0
    )

    q = np.array([-3 * dq, -2 * dq, -dq, 0.0, dq, 2 * dq, 3 * dq])
    fvals = corradini_pz(rs, q)
    f6 = (
        (
            -fvals[0]
            + 6 * fvals[1]
            - 15 * fvals[2]
            + 20 * fvals[3]
            - 15 * fvals[4]
            + 6 * fvals[5]
            - fvals[6]
        )
        / dq**6
        / 720.0
    )

    return f0, f2, f4, f6


def pi_moment_kappa(n, rs):
    """Analytic moments of Pi(q) = chi0/(1 - chi0*(vc_kappa + fxc)) with kappa."""
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    f0, f2, f4, f6 = get_fxc_derivatives(rs)
    pi = np.pi
    K = kappa  # shorthand

    # Common denominator base: D = 4*kF*pi + (f0*kF + pi^2)*K^2
    D = 4 * kF * pi + (f0 * kF + pi**2) * K**2

    if n == 0:
        return -kF * K**2 / (4 * pi * D)

    if n == 1:
        num = -(pi**2 * K**4 + 12 * kF**3 * (-4 * pi + f2 * K**4))
        return num / (8 * kF * pi * D**2)

    if n == 2:
        num = (
            2880 * kF**5 * pi * (f0 * kF + pi**2)
            + 480 * kF**3 * pi * (12 * f2 * kF**3 + pi**2) * K**2
            + 32 * kF * pi * (90 * f4 * kF**5 + pi**2) * K**4
            + (
                -720 * (f2**2 - f0 * f4) * kF**6
                + 8 * kF * (f0 - 15 * f2 * kF**2 + 90 * f4 * kF**4) * pi**2
                + 3 * pi**4
            )
            * K**6
        )
        return num / (24 * kF**3 * pi * D**3)

    if n == 3:
        num = (
            80640
            * kF**6
            * pi
            * (
                -3 * f0**2 * kF**3
                + 12 * f2 * kF**3 * pi
                - 6 * f0 * kF**2 * pi**2
                + pi**3
                - 3 * kF * pi**4
            )
            - 2688
            * kF**4
            * pi
            * (
                180 * f0 * f2 * kF**5
                - 720 * f4 * kF**5 * pi
                + 15 * kF**2 * (f0 + 12 * f2 * kF**2) * pi**2
                - 8 * pi**3
                + 15 * kF * pi**4
            )
            * K**2
            + 48
            * kF**2
            * pi
            * (
                5040 * (-3 * f2**2 + 2 * f0 * f4) * kF**7
                + 20160 * f6 * kF**7 * pi
                + 56 * kF**2 * (2 * f0 - 45 * kF**2 * (f2 - 4 * f4 * kF**2)) * pi**2
                + 29 * pi**3
                + 7 * kF * pi**4
            )
            * K**4
            + 8
            * kF
            * pi
            * (
                60480 * (-f2 * f4 + f0 * f6) * kF**8
                + 3
                * kF
                * (
                    29 * f0
                    - 112 * (2 * f2 * kF**2 + 15 * kF**4 * (f4 - 12 * f6 * kF**2))
                )
                * pi**2
                + 31 * pi**4
            )
            * K**6
            + (
                60480 * (f2**3 - 2 * f0 * f2 * f4 + f0**2 * f6) * kF**9
                + 3
                * kF**2
                * (
                    29 * f0**2
                    + 5040 * f2 * kF**4 * (f2 - 8 * f4 * kF**2)
                    - 224 * f0 * (2 * f2 * kF**2 + 15 * kF**4 * (f4 - 12 * f6 * kF**2))
                )
                * pi**2
                + 2
                * kF
                * (31 * f0 - 42 * kF**2 * (f2 + 120 * kF**2 * (f4 - 6 * f6 * kF**2)))
                * pi**4
                + 10 * pi**6
            )
            * K**8
        )
        return -num / (48 * kF**5 * pi * D**4)

    raise ValueError("n must be 0,1,2,3")


def pi_moment_numerical(n, rs, q_max=200, N=500000):
    """Compute C_n numerically from Pi(q) via Taylor coefficients."""
    kF = (9 * np.pi / 4) ** (1 / 3) / rs

    # Dense q grid for numerical derivative
    dq = q_max / N
    q = np.arange(1, N + 1) * dq

    chi0q = chi00q(q, rs)
    fxc = corradini_pz(rs, q)
    vc = 4 * np.pi / (q**2 + kappa**2)
    piq = chi0q / (1 - chi0q * (vc + fxc))

    # C_n = (-1)^n * (2n+1)/(4pi) * Pi^(2n)(0)
    # Pi^(2n)(0) = (2n)! * [q^(2n) coefficient of Pi(q) Taylor series]
    # Numerically: fit even polynomial to Pi(q) for small q

    # Use small-q region
    q_fit = q[q < 0.5 * kF]
    pi_fit = piq[q < 0.5 * kF]

    # Fit even polynomial: Pi(q) = a0 + a2*q^2 + a4*q^4 + a6*q^6
    # Using q^2 as variable
    q2 = q_fit**2
    # Fit polynomial in q^2 of degree 3
    coeffs = np.polyfit(q2, pi_fit, 3)  # coeffs[0]*q^6 + ... + coeffs[3]
    # coeffs = [a6, a4, a2, a0]
    a0 = coeffs[3]  # Pi(0)
    a2 = coeffs[2]  # coefficient of q^2
    a4 = coeffs[1]  # coefficient of q^4
    a6 = coeffs[0]  # coefficient of q^6

    # Pi^(0)(0) = a0, so Pi_hat(0) = a0
    # Pi^(2)(0) = 2!*a2, so Pi_hat(1) = 2!*a2
    # Pi^(4)(0) = 4!*a4, so Pi_hat(2) = 4!*a4
    # Pi^(6)(0) = 6!*a6, so Pi_hat(3) = 6!*a6

    factorials = [1, 2, 24, 720]  # 0!, 2!, 4!, 6!
    a = [a0, a2, a4, a6]

    pi_hat_n = factorials[n] * a[n]
    C_n = (-1) ** n * (2 * n + 1) / (4 * np.pi) * pi_hat_n
    return C_n


# Test for several rs values
print(
    f"{'rs':>6s} | {'n':>2s} | {'C_n(analytic)':>16s} | {'C_n(numerical)':>16s} | {'rel_err':>10s}"
)
print("-" * 70)

for rs in [1.0, 2.0, 4.0, 6.0, 8.0]:
    for n in range(4):
        c_ana = pi_moment_kappa(n, rs)
        c_num = pi_moment_numerical(n, rs)
        rel = abs(c_ana - c_num) / (abs(c_ana) + 1e-30)
        print(f"{rs:6.2f} | {n:2d} | {c_ana:16.8e} | {c_num:16.8e} | {rel:10.2e}")
    print()

# Also compare old (no-kappa) pi_moment with kappa version
print("\n=== Comparison: old pi_moment (no kappa) vs new (with kappa) ===")
from utils.physics import pi_moment as pi_moment_old

for rs in [1.0, 4.0]:
    for n in range(4):
        old = pi_moment_old(n, rs)
        new = pi_moment_kappa(n, rs)
        print(
            f"rs={rs}, n={n}: old={old:.8e}, new={new:.8e}, diff={abs(new - old) / abs(old + 1e-30) * 100:.2f}%"
        )
    print()
