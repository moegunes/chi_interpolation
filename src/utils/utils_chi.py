import numpy as np
from numpy import pi, sqrt

from utils.fourier import chi_r_from_chi_q_fast


def get_chi(q, rs):
    chi0q = chi00q(q, rs)  # -chi00q(q,rs,interacting=False)[0]
    fxc = corradini_pz(rs, q)
    vc = 4 * np.pi / q**2

    chiq = chi0q / (1 - chi0q * (vc + fxc))
    chiR = chi_r_from_chi_q_fast(q, chiq)[1]

    return chiR


def chi00q(q, rs):  # one spin
    "Lindhard function (reciprocal space) from Vignale, eqn. 18 in the report"
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    q = q + 1e-10  # q*filter_high +  1e-10*filter_low
    Q = q / (kF)
    res = (
        -kF / (2 * np.pi**2) * (1 - (Q / 4 - 1 / Q) * np.log(np.abs((Q + 2) / (Q - 2))))
    )
    return res


def p_correlation_PZ(n):
    rs = (4 * np.pi / 3 * n) ** (-1 / 3)
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    ######
    Au = 0.0311
    Bu = -0.048
    Cu = 0.0020
    Du = -0.0116
    filterlow = rs < 1
    filterhigh = rs >= 1
    reslow = (
        Au * np.log(rs)
        + (Bu - 1 / 3 * Au)
        + 2 / 3 * Cu * rs * np.log(rs)
        + 1 / 3 * (2 * Du - Cu) * rs
    )
    v_cep_rs = gamma / (1 + beta1 * np.sqrt(rs) + beta2 * rs)
    reshigh = (
        v_cep_rs
        * (1 + 7 / 6 * beta1 * np.sqrt(rs) + 4 / 3 * beta2 * rs)
        / (1 + beta1 * np.sqrt(rs) + beta2 * rs)
    )
    return reslow * filterlow + reshigh * filterhigh


##corradini PZ
def diffv_cep(r_s):
    # Multiplied v_cep from above with r_s and differentiated wrt r_s
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    # res = gamma * (beta1 * np.sqrt(r_s) + 2) / \
    #    (2 * (beta1 * np.sqrt(r_s) + beta2 * r_s + 1) ** 2)
    res = (beta1 * gamma * np.sqrt(r_s)) / (
        2 * (beta1 * np.sqrt(r_s) + beta2 * r_s + 1) ** 2
    ) + gamma / (beta1 * np.sqrt(r_s) + beta2 * r_s + 1) ** 2
    return res


def diffvc(rho):
    # from dp-code
    third = 1.0 / 3.0
    a = 0.0311
    # b = -0.0480  #  never used?
    c = 0.0020
    d = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    r_s = (3.0 / (4.0 * np.pi * rho)) ** third

    stor1 = (1.0 + beta1 * np.sqrt(r_s) + beta2 * r_s) ** (-3.0)
    stor2 = (
        -0.41666667 * beta1 * (r_s ** (-0.5))
        - 0.5833333 * (beta1**2)
        - 0.66666667 * beta2
    )
    stor3 = -1.75 * beta1 * beta2 * np.sqrt(r_s) - 1.3333333 * r_s * (beta2**2)
    reshigh = gamma * stor1 * (stor2 + stor3)
    reslow = a / r_s + 0.66666667 * (c * np.log(r_s) + d) + 0.33333333 * c

    reshigh = reshigh * (-4.0 * np.pi / 9.0) * (r_s**4)
    reslow = reslow * (-4.0 * np.pi / 9.0) * (r_s**4)

    filterlow = r_s < 1
    filterhigh = r_s >= 1
    return reslow * filterlow + reshigh * filterhigh


def corradini_pz(r_s, q):
    q = q + 1e-18
    # Q is given in multiples of k_F (Q = q / k_F)
    rho_avg = 1.0 / (r_s**3.0 * 4.0 * np.pi / 3.0)
    k_F = (3.0 * np.pi**2.0 * rho_avg) ** (1.0 / 3.0)
    Q = q / k_F
    # Q = q
    e = 1  # atomic units
    #   diff_mu = 1                 # How to model this for HEG?
    #                                 This should be d \mu_c / d n_0
    diff_mu = diffvc(rho_avg)
    A = 1.0 / 4.0 - (k_F**2.0) / (4.0 * np.pi * e**2.0) * diff_mu
    #   diff_rse = 1                # How to model this for HEG? e_c(r_s) !!
    #                                 This should be d(r_s * e_c) / d r_s
    diff_rse = diffv_cep(r_s)
    C = np.pi / (2.0 * e**2.0 * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    x = r_s ** (1.0 / 2.0)
    B = (1.0 + a1 * x + a2 * x**3.0) / (3.0 + b1 * x + b2 * x**3.0)
    g = B / (A - C)
    alpha = 1.5 / (r_s ** (1.0 / 4.0)) * A / (B * g)
    beta = 1.2 / (B * g)
    Gcor = (
        C * Q**2.0
        + (B * Q**2.0) / (g + Q**2.0)
        + alpha * Q**4.0 * np.exp(-beta * Q**2.0)
    )
    return -4.0 * np.pi * e**2.0 / (q**2.0) * Gcor


def xc_real(nx):
    # nx=np.sqrt(nx.real**2)
    potential = -((3.0 / np.pi) ** (1.0 / 3.0)) * nx ** (1 / 3) + p_correlation_PZ(
        nx
    )  ##the potential in the direct space
    return potential


def fxc_lda_scalar(rs):
    a = 0.0311
    b = -0.0480
    c = 0.0020
    d = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    rho = 4 * pi / 3 * rs**3
    # rho=0.1
    # rs=(3.0/(4.0*pi*rho))**athird
    # rs=8
    # two different calculations depending if Rs <> 1
    if rs >= 1:
        stor1 = (1.0 + beta1 * sqrt(rs) + beta2 * rs) ** (-3.0)
        stor2 = (
            -0.41666667 * beta1 * (rs ** (-0.5))
            - 0.5833333 * (beta1**2)
            - 0.66666667 * beta2
        )
        stor3 = -1.75 * beta1 * beta2 * sqrt(rs) - 1.3333333 * rs * (beta2**2)
        diffvc = gamma * stor1 * (stor2 + stor3)
    else:
        diffvc = a / rs + 0.66666667 * (c * np.log(rs) + d) + 0.33333333 * c

    diffvc = diffvc * (-4.0 * pi / 9.0) * (rs**4)

    ##### diffVx
    # real rho,rs,b,bb,bb1
    # real rel, vxnr, difrel
    vxnr = -((3 * rho / pi) ** 0.3333333333)
    b = 0.0140 / rs
    rel = -0.5 + 1.5 * np.log(b + sqrt(1.0 + b * b)) / (b * sqrt(1.0 + b * b))
    bb = b * b
    bb1 = 1.0 + bb
    difrel = (1.5 / (b * bb1)) - 1.5 * np.log(b + sqrt(bb1)) * (1.0 + 2.0 * bb) * (
        bb1 ** (-1.5)
    ) / bb
    difrel = difrel * (-0.0140) / (rs * rs)
    diffvx = (0.610887057 / (rs * rs)) * rel + vxnr * difrel
    diffvx = diffvx * (-4.0 * pi / 9.0) * (rs**4)
    return diffvc + diffvx


def G_Moroni(rs, q, n=8):
    q = q + 1e-18
    # Q is given in multiples of k_F (Q = q / k_F)
    rho_avg = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    k_F = (3.0 * np.pi**2.0 * rho_avg) ** (1.0 / 3.0)
    Q = q / k_F
    # Q = q
    e = 1  # atomic units
    #   diff_mu = 1                 # How to model this for HEG?
    #                                 This should be d \mu_c / d n_0
    diff_mu = diffvc(rho_avg)
    A = 1.0 / 4.0 - (k_F**2.0) / (4.0 * np.pi * e**2.0) * diff_mu
    #   diff_rse = 1                # How to model this for HEG? e_c(r_s) !!
    #                                 This should be d(r_s * e_c) / d r_s
    diff_rse = diffv_cep(rs)
    C = np.pi / (2.0 * e**2.0 * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    # n=8
    x = rs ** (1.0 / 2.0)
    B = (1.0 + a1 * x + a2 * x**3.0) / (3.0 + b1 * x + b2 * x**3.0)
    G = (((A - C) ** (-n) + (Q**2 / B) ** n) ** (-1 / n) + C) * Q**2
    return G
