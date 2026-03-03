import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from optimization.production import get_pi_interp
from utils.fourier import chi_q_from_chi_r_fast
from utils.physics import canon_cos_phase
from utils.utils_chi import chi00q, corradini_pz, get_gas_params, get_pi, get_piq


def _safe_tight_layout(fig, **kwargs):
    """Apply tight_layout robustly, with a fallback when LaTeX rendering fails."""
    try:
        fig.tight_layout(**kwargs)
    except RuntimeError:
        fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.35)


def plot_parameters(params_dict):
    keys = list(params_dict.keys())
    rsl = [k for k in keys if isinstance(k, float)]
    rsl.sort()
    # rsl = rsl[7:]
    font_size = 6
    marker_size = 2
    lww = 0.7
    plt.rcParams["figure.dpi"] = 600
    # figsize=(5.92, 10.18)
    fig, ax = plt.subplots(nrows=4, ncols=1, constrained_layout=True, sharex=True)

    B0l = []
    B1l = []
    for j in range(3):
        coef1l = []
        coef2l = []

        for rs_i in range(len(rsl)):
            rs = rsl[rs_i]
            mode1 = params_dict[rs][0:3]
            mode2 = params_dict[rs][3:6]
            coef1, coef2 = mode1[j], mode2[j]
            if j == 2:
                coef1l.append(canon_cos_phase(np.mod(coef1, 2 * np.pi)))
                coef2l.append(canon_cos_phase(np.mod(coef2, 2 * np.pi)))
            else:
                coef1l.append(coef1)
                coef2l.append(coef2)

        ax[j].plot(rsl, coef1l, "k-o", label=r"$m=1$", lw=lww, markersize=marker_size)
        ax[j].plot(rsl, coef2l, "r-o", label=r"$m=2$", lw=lww, markersize=marker_size)

        # ax[j].set_xlabel(r'$r_s$',fontsize=font_size)
        ax[j].legend(fontsize=font_size)
        ax[j].tick_params(axis="both", labelsize=font_size)
        if j == 0:
            ax[j].set_ylabel(r"$\alpha_m$", fontsize=font_size)
        elif j == 1:
            ax[j].set_ylabel(r"$k_m$", fontsize=font_size)
        elif j == 2:
            ax[j].set_ylabel(r"$\phi_m$", fontsize=font_size)

    B0l, B1l = get_constraints(params_dict, rsl)

    ax[3].plot(rsl, B0l, "k-o", label=r"$m=1$", lw=lww, markersize=marker_size)
    ax[3].plot(rsl, B1l, "r-o", label=r"$m=2$", lw=lww, markersize=marker_size)
    # ax[3].set_xlabel(r'$r_s$',fontsize=font_size)
    ax[3].legend(fontsize=font_size)
    ax[3].set_ylabel(r"$A_m$", fontsize=font_size)
    ax[3].tick_params(axis="both", labelsize=font_size)
    # plt.plot(rsl,0*np.array(rsl)+np.pi/2)
    # plt.ylim(-.15,.15)
    fig.supxlabel(r"$r_s$", fontsize=font_size, y=0.04)
    fig.subplots_adjust(hspace=0.05, left=0.15, right=0.95, top=0.92, bottom=0.12)
    plt.legend(fontsize=font_size)
    plt.savefig("parameters_best.png", bbox_inches="tight", dpi=600)


def plot_chi(r, q, params_dict, rs, error=False):
    from matplotlib.ticker import ScalarFormatter

    kF, n0, NF = get_gas_params(rs)
    piR = get_pi(q, rs)
    pi_interpp = get_pi_interp(r, q, params_dict, rs)
    dr = r[1] - r[0]
    font_size = 8

    fig, ax = plt.subplots(nrows=2, ncols=1)

    fxc = corradini_pz(rs, q)
    vc = 4 * np.pi / q**2

    chi0q = chi00q(q, rs)
    piq = chi0q / (1 - chi0q * fxc)

    FT_q, FT_chiq = chi_q_from_chi_r_fast(r, pi_interpp, qlist=None)

    if error:
        # ax[0].plot(kF*r[::10],chiR[::10]/NF,'k-',label=r'$\chi^h(r)$')
        ax[0].plot(
            kF * r[100::180],
            (piR - pi_interpp)[100::180] / (2 * kF**4 / pi**3),
            "k-",
            markersize=1,
            label=r"$\Pi_{M=2}^{h,\mathrm{interp.}}(r)$",
        )
        # ax[0].plot(kF*r[::180],np.abs((chiR-chi_interp)/chiR)[::180]*100,'ro',markersize=1,label=r'$\chi_{M=2}^{h,\mathrm{interp.}}(r)$')
        ax[0].plot(kF * r, 0 * kF * r, "k", lw=0.25)
        lim_upper = 0.0002  # max(chiR/(2*kF**4/pi**3))*.2
        ax[0].set_ylim(-lim_upper / 1, lim_upper)
        ax[0].set_xlim(0, 24)
        ax[0].set_xlabel(r"$k_F r$", fontsize=font_size, labelpad=2)
        ax[0].set_ylabel(r"$\Delta\Pi(r)/6\pi n_0 N_\mathrm{F}$", fontsize=font_size)
        # ax[0].set_title(fr'$r_s = {rs}$',fontsize=font_size)

        # ax[1].plot(FT_q/kF,-np.abs(FT_chiq-chiq)/chiq*100,'r-.',label=r' $\chi^0(q)$ invFT',lw=1)
        ax[1].plot(FT_q / kF, (piq - FT_chiq) / NF, "k-", lw=1)

        # error_qmc = 0.2577 / 2
        limy = max(abs(piq - FT_chiq) / NF) * 1.2  # error_qmc*2
        # plt.axhspan(-error_qmc,error_qmc, xmax=10,color='grey', alpha=0.3)
        ax[1].set_ylim(-limy, limy)
        ax[1].plot(FT_q / kF, 0 * FT_q, "k", lw=0.25)
        # ax[1].set_ylim(0,10)
        ax[1].set_xlim(0, 10)
        ax[1].set_xlabel(r"$q/k_F$", fontsize=font_size, labelpad=2)
        ax[1].set_ylabel(r"$\Delta\Pi(q)/N_\mathrm{F}$", fontsize=font_size)
        # ax[1].set_title(fr'$r_s = {rs}$')
        # ax[1].legend(fontsize=font_size)
    else:
        ax[0].plot(
            kF * r[::10], piR[::10] / (2 * kF**4 / pi**3), "k-", label=r"$\Pi^h(r)$"
        )
        ax[0].plot(
            kF * r[::180],
            pi_interpp[::180] / (2 * kF**4 / pi**3),
            "r-.",
            markersize=1,
            label=r"$\Pi_{M=2}^{h,\mathrm{interp.}}(r)$",
        )
        ax[0].plot(kF * r, 0 * kF * r, "k", lw=0.5)
        lim_upper = max(piR / (2 * kF**4 / pi**3)) * 1.2
        ax[0].set_ylim(-lim_upper / 1, lim_upper)
        ax[0].set_xlim(0, 12)
        ax[0].set_xlabel(r"$k_F r$", fontsize=font_size, labelpad=2)
        ax[0].set_ylabel(r"$\Pi(r)/6\pi n_0 N_\mathrm{F}$", fontsize=font_size)
        # ax[0].set_title(fr'$r_s = {rs}$',fontsize=font_size)
        ax[0].legend(loc="lower right", fontsize=font_size)
        ax[0].tick_params(axis="both", labelsize=font_size)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, -1))  # force ×10^{-1}

        ax[0].yaxis.set_major_formatter(formatter)
        ax[0].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1))

        ax[1].plot(q / kF, -piq / NF, "k", label=r" $\Pi(q)$ analytical", lw=1)
        ax[1].plot(FT_q / kF, -FT_chiq / NF, "r-.", label=r" $\Pi(q)$ invFT", lw=1)
        ax[1].set_xlim(0, 10)
        ax[1].set_xlabel(r"$q/k_F$", fontsize=font_size, labelpad=2)
        ax[1].set_ylabel(r"$-\Pi(q)/N_\mathrm{F}$", fontsize=font_size)
        # ax[1].set_title(fr'$r_s = {rs}$')
        ax[1].legend(fontsize=font_size)
        ax[1].tick_params(axis="both", labelsize=font_size)

    print(f"∫Pi(r)r^2dr: {np.sum(pi_interpp * r**2) * dr:.6f}")
    # fig.subplots_adjust(hspace=1.4)
    fig.subplots_adjust(hspace=0.35, left=0.15, right=0.95, top=0.92, bottom=0.12)
    plt.savefig(f"delta_pi-rs{rs}-error{error}.png", bbox_inches="tight", dpi=600)


def get_constraints(params_dict, rslist):
    model = params_dict["model"]
    B0_list = []
    B1_list = []

    for rs in rslist:
        p_opt = params_dict[rs]
        B0, B1 = model(
            r=0, rs=rs, params=p_opt, get_constraints=True
        )  # r=0 is a dummy value
        B0_list.append(B0)
        B1_list.append(B1)
    return np.array(B0_list), np.array(B1_list)


# ===== Parametric forms for interpolation =====
def f_denom(x, a, b, c):
    """A / (rs**B + C) - user form"""
    return a / (x**b + c)


def f_two_denom(x, a, b, c, d, e, f):
    """A / (rs**B + C) + D / (rs**E + F) - user form"""
    return a / (x**b + c) + d / (x**e + f)


def f_superpose(x, a, b, c, d):
    """A*rs**B + C*rs**D - user form"""
    return a * x**b + c * x**d


def f_power(x, a, b, c):
    """a + b*rs^c"""
    return a + b * x**c


def f_poly2(x, a, b, c):
    """a + b*rs + c*rs^2"""
    return a + b * x + c * x**2


def f_poly3(x, a, b, c, d):
    """a + b*rs + c*rs^2 + d*rs^3"""
    return a + b * x + c * x**2 + d * x**3


def f_pade11(x, a, b, c):
    """(a + b*rs)/(1 + c*rs)"""
    return (a + b * x) / (1 + c * x)


def f_two_pade11(x, a, b, c, d, e, f):
    """(a + b√rs)/(1 + c√rs) + (d + e√rs)/(1 + f√rs) — two Padé[1/1] in √rs (VWN/PZ variable), 6 params"""
    return (a + b * x) / (1 + c * x) + (d + e * x) / (1 + f * x)


def f_pade22(x, a, b, c, d, e):
    """(a + b*rs + c*rs^2)/(1 + d*rs + e*rs^2)"""
    return (a + b * x + c * x**2) / (1 + d * x + e * x**2)


def f_inv(x, a, b, c):
    """a + b/(rs + c)"""
    return a + b / (x + c)


def f_sat(x, a, b, c):
    """a * (1 - exp(-b*rs)) + c"""
    return a * (1 - np.exp(-b * x)) + c


def f_dblpow(x, a, b, c, d, e):
    """a + b*rs^c + d*rs^e"""
    with np.errstate(invalid="ignore"):
        return a + b * np.abs(x) ** c + d * np.abs(x) ** e


def f_shifted_pade12(x, a, c0, c1, d1, d2):
    """a + (c0 + c1*rs)/(1 + d1*rs + d2*rs^2)  —  shifted Padé [1,2]
    Saturates to `a` at large rs; bump from rational correction."""
    return a + (c0 + c1 * x) / (1 + d1 * x + d2 * x**2)


def f_shifted_pade12_sqrt(x, a, c0, c1, d1, d2):
    """a + (c0 + c1*sqrt(rs))/(1 + d1*sqrt(rs) + d2*rs)  —  sqrt variant
    Better small-rs scaling (Perdew-Wang style)."""
    s = np.sqrt(np.abs(x))
    return a + (c0 + c1 * s) / (1 + d1 * s + d2 * x)


# ----- HEG / QMC-literature-motivated forms -----


def f_pade11_sqrt(x, a, b, c):
    """(a + b√rs)/(1 + c√rs) — Padé[1/1] in √rs (VWN/PZ variable), 3 params"""
    s = np.sqrt(np.abs(x))
    return (a + b * s) / (1 + c * s)


def f_two_pade11_sqrt(x, a, b, c, d, e, f):
    """(a + b√rs)/(1 + c√rs) + (d + e√rs)/(1 + f√rs) — two Padé[1/1] in √rs (VWN/PZ variable), 6 params"""
    s = np.sqrt(np.abs(x))
    return (a + b * s) / (1 + c * s) + (d + e * s) / (1 + f * s)


def f_pade21_sqrt(x, a, b, c, d):
    """(a + b√rs + c·rs)/(1 + d√rs) — Padé[2/1] in √rs, 4 params"""
    s = np.sqrt(np.abs(x))
    return (a + b * s + c * s**2) / (1 + d * s)


def f_pade22_sqrt(x, a, b, c, d, e):
    """(a + b√rs + c·rs)/(1 + d√rs + e·rs) — Padé[2/2] in √rs, 5 params.
    Standard QMC interpolation form (Ceperley-Alder / VWN class)."""
    s = np.sqrt(np.abs(x))
    return (a + b * s + c * s**2) / (1 + d * s + e * s**2)


def f_pade23_sqrt(x, a, b, c, d, e, f, g):
    """(a + b√rs + c·rs )/(1 + d√rs + e·rs + f·rs^3/2) — Padé[2/3] in √rs, 7 params.
    Standard QMC interpolation form (Ceperley-Alder / VWN class)."""
    s = np.sqrt(np.abs(x))
    return g + (a + b * s + c * s**2) / (1 + d * s + e * s**2 + f * s**3)


def f_pade23(x, a, b, c, d, e, f, g):
    """(a + b√rs + c·rs )/(1 + d√rs + e·rs + f·rs^3/2) — Padé[2/3] in √rs, 7 params.
    Standard QMC interpolation form (Ceperley-Alder / VWN class)."""
    return g + (a + b * x + c * x**2) / (1 + d * x + e * x**2 + f * x**3)


def f2_pade23_sqrt(x, a, b, c, d, e, f, g, h):
    """(a + b√rs + c·rs )/(h + d√rs + e·rs + f·rs^3/2) — Padé[2/3] in √rs, 8 params.
    Standard QMC interpolation form (Ceperley-Alder / VWN class)."""
    s = np.sqrt(np.abs(x))
    return g + (a + b * s + c * s**2 + h * s**3) / (1 + d * s + e * s**2 + f * s**3)


def f2_pade23(x, a, b, c, d, e, f, g, h):
    """(a + b√rs + c·rs )/(h + d√rs + e·rs + f·rs^3/2) — Padé[2/3] in √rs, 8 params.
    Standard QMC interpolation form (Ceperley-Alder / VWN class)."""
    return g + (a + b * x + c * x**2 + h * x**3) / (1 + d * x + e * x**2 + f * x**3)


def f_shifted_pz(x, a, b, c, d):
    """a + b/(1 + c√rs + d·rs) — shifted Perdew-Zunger form, 4 params.
    Saturates to `a` at large rs; Coulomb-hole correction at small rs."""
    return a + b / (1 + c * np.sqrt(np.abs(x)) + d * x)


def f_sat_bump(x, a, b, c, d):
    """a + (b + c·rs)·exp(-d·rs) — saturation + exponential bump, 4 params.
    f(0)=a+b, f(∞)=a; bump at rs=1/d - b/c."""
    return a + (b + c * x) * np.exp(-d * x)


PARAMETRIC_FORMS = {
    "denom": (f_denom, 3),
    "superpose": (f_superpose, 4),
    "a+b*rs^c": (f_power, 3),
    "poly2": (f_poly2, 3),
    "poly3": (f_poly3, 4),
    "Pade[1/1]": (f_pade11, 3),
    "Pade[2/2]": (f_pade22, 5),
    "a+b/(rs+c)": (f_inv, 3),
    "saturating": (f_sat, 3),
    "a+b*rs^c+d*rs^e": (f_dblpow, 5),
    "sPade[1/2]": (f_shifted_pade12, 5),
    "sPade[1/2]sqrt": (f_shifted_pade12_sqrt, 5),
    # HEG / QMC literature forms
    "PZ[1/1]√": (f_pade11_sqrt, 3),
    "PZ[2/1]√": (f_pade21_sqrt, 4),
    "PZ[2/2]√": (f_pade22_sqrt, 5),
    "sPZ": (f_shifted_pz, 4),
    "sat+bump": (f_sat_bump, 4),
    "PZ[2/3]√": (f_pade23_sqrt, 7),
    "PZ[2/3]": (f_pade23, 7),
    "mPZ[2/3]√": (f2_pade23_sqrt, 8),
    "mPZ[2/3]": (f2_pade23, 8),
    "two Pade[1/1]": (f_two_pade11, 6),
    "two Pade[1/1]√": (f_two_pade11_sqrt, 6),
    "two denom": (f_two_denom, 6),
}


def fit_parametric_forms(rs_arr, y_arr, forms=None):
    """Fit all parametric forms to data.

    Returns dict of {form_name: (func, ncoeff, popt, rmse, maxerr, maxpct)}
    """
    from scipy.optimize import curve_fit

    if forms is None:
        forms = PARAMETRIC_FORMS

    rng = np.max(y_arr) - np.min(y_arr)
    results = {}

    for form_name, (func, ncoeff) in forms.items():
        try:
            popt, _ = curve_fit(func, rs_arr, y_arr, maxfev=50000)
            pred = func(rs_arr, *popt)
            residuals = np.abs(pred - y_arr)
            rmse = np.sqrt(np.mean(residuals**2))
            maxerr = np.max(residuals)
            maxpct = maxerr / (rng + 1e-30) * 100
            results[form_name] = (func, ncoeff, popt, rmse, maxerr, maxpct)
        except Exception:
            pass

    return results


def plot_parameter_fits(params_dict, param_idx=0, forms_to_plot=None, ax=None):
    """Plot a single parameter vs rs with fitted curves overlaid.

    param_idx: 0=alpha0, 1=f0, 2=phi0, 3=alpha1, 4=f1, 5=phi1, 6=B0, 7=B1
    """
    from utils.physics import J_n_m_kFr, delta_C

    param_names = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1", "B0", "B1"]
    param_labels = [
        r"$\alpha_0$",
        r"$f_0$",
        r"$\phi_0$",
        r"$\alpha_1$",
        r"$f_1$",
        r"$\phi_1$",
        r"$B_0$",
        r"$B_1$",
    ]

    # Extract data
    rsl = sorted(
        [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
    )
    rs_arr = np.array(rsl)
    data6 = np.array([params_dict[rs] for rs in rsl])

    if param_idx < 6:
        y_arr = data6[:, param_idx]
    else:
        # Compute B0 or B1
        B0_arr, B1_arr = [], []
        for r_val in rsl:
            p = params_dict[r_val]
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
        y_arr = np.array(B0_arr if param_idx == 6 else B1_arr)

    # Fit forms
    fit_results = fit_parametric_forms(rs_arr, y_arr)

    # Sort by RMSE
    sorted_forms = sorted(fit_results.items(), key=lambda x: x[1][3])

    if forms_to_plot is None:
        forms_to_plot = [name for name, _ in sorted_forms[:4]]  # top 4

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(rs_arr, y_arr, "ko", markersize=5, label="Data", zorder=10)

    rs_dense = np.linspace(rs_arr.min(), rs_arr.max(), 200)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, form_name in enumerate(forms_to_plot):
        if form_name in fit_results:
            func, ncoeff, popt, rmse, maxerr, maxpct = fit_results[form_name]
            y_fit = func(rs_dense, *popt)
            label = rf"{form_name} ({ncoeff}p, {maxpct:.1f}%)"
            ax.plot(rs_dense, y_fit, "-", color=colors[i], lw=1.5, label=label)

    ax.set_xlabel(r"$r_s$", fontsize=12)
    ax.set_ylabel(param_labels[param_idx], fontsize=12)
    ax.set_title(f"{param_names[param_idx]} parametric fits", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    return fit_results


def plot_all_parameter_fits(params_dict, forms_to_plot=None, figsize=(14, 10)):
    """Plot all 8 parameters with their best fits in a 4x2 grid."""
    with plt.rc_context({"text.usetex": False}):
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        axes = axes.flatten()

        for i in range(8):
            plot_parameter_fits(
                params_dict, param_idx=i, forms_to_plot=forms_to_plot, ax=axes[i]
            )

        _safe_tight_layout(fig)
    return fig


def plot_fit_residuals(params_dict, param_idx=0, form_name="Pade[2/2]", ax=None):
    """Plot residuals of a parametric fit."""
    from utils.physics import J_n_m_kFr, delta_C

    param_names = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1", "B0", "B1"]

    # Extract data
    rsl = sorted(
        [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
    )
    rs_arr = np.array(rsl)
    data6 = np.array([params_dict[rs] for rs in rsl])

    if param_idx < 6:
        y_arr = data6[:, param_idx]
    else:
        B0_arr, B1_arr = [], []
        for r_val in rsl:
            p = params_dict[r_val]
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
        y_arr = np.array(B0_arr if param_idx == 6 else B1_arr)

    # Fit the requested form
    fit_results = fit_parametric_forms(
        rs_arr, y_arr, {form_name: PARAMETRIC_FORMS[form_name]}
    )

    if form_name not in fit_results:
        raise ValueError(f"Form {form_name} failed to fit")

    func, ncoeff, popt, rmse, maxerr, maxpct = fit_results[form_name]
    y_pred = func(rs_arr, *popt)
    residuals = y_arr - y_pred

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(rs_arr, residuals, width=0.08, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$r_s$", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.set_title(
        f"{param_names[param_idx]} residuals ({form_name}, RMSE={rmse:.2e})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="y")

    return residuals, popt


def fit_all_parameters(params_dict, form_name="Pade[2/2]"):
    """Fit parametric form(s) to all 6 nonlinear parameters.

    Parameters
    ----------
    params_dict : dict
        Parameter dictionary keyed by rs values.
    form_name : str or dict
        If str, use the same form for all 6 parameters.
        If dict, map parameter names to form names, e.g.
            {"alpha0": "Pade[2/2]", "f0": "sPade[1/2]sqrt", ...}
        Missing keys fall back to "Pade[2/2]".

    Returns dict with fitted coefficients for each parameter.
    """
    from scipy.optimize import curve_fit

    rsl = sorted(
        [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
    )
    rs_arr = np.array(rsl)
    data6 = np.array([params_dict[rs] for rs in rsl])

    param_names = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]

    # Normalise form_name to a per-parameter dict
    if isinstance(form_name, str):
        form_map = {p: form_name for p in param_names}
        label = form_name
    elif isinstance(form_name, dict):
        # allow a special key '_default' for fallback
        default = form_name.get("_default", "Pade[2/2]")
        form_map = {p: form_name.get(p, default) for p in param_names}
        label = "mixed"
    else:
        raise TypeError("form_name must be a string or a dict mapping parameter->form")

    # Validate requested forms
    for p, fname in form_map.items():
        if not isinstance(fname, str):
            raise TypeError(
                f"Form name for parameter '{p}' must be a string, got {type(fname)}"
            )
        if fname not in PARAMETRIC_FORMS:
            raise ValueError(
                f"Requested form '{fname}' for parameter '{p}' is not in PARAMETRIC_FORMS"
            )

    fits = {"form_name": label, "form_map": form_map, "rs_arr": rs_arr}

    for i, pname in enumerate(param_names):
        fname = form_map[pname]
        func, ncoeff = PARAMETRIC_FORMS[fname]
        y_arr = data6[:, i]
        try:
            popt, _ = curve_fit(func, rs_arr, y_arr, maxfev=50000)
            pred = func(rs_arr, *popt)
            rmse = np.sqrt(np.mean((pred - y_arr) ** 2))
            maxpct = (
                np.max(np.abs(pred - y_arr))
                / (np.max(y_arr) - np.min(y_arr) + 1e-30)
                * 100
            )
            fits[pname] = {
                "popt": popt,
                "rmse": rmse,
                "maxpct": maxpct,
                "func": func,
                "form_name": fname,
            }
        except Exception as e:
            fits[pname] = {"popt": None, "error": str(e)}

    return fits


def get_interpolated_params(rs, fits):
    """Get 6 interpolated parameters at a given rs using fitted forms.

    Supports per-parameter forms (each entry in fits stores its own func).
    """
    params = []
    for pname in ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]:
        entry = fits[pname]
        if entry["popt"] is None:
            raise ValueError(f"No fit available for {pname}")
        # Per-parameter func stored by new fit_all_parameters
        func = entry.get("func") or fits.get("func")
        if func is None:
            # Legacy fallback: single form_name
            func = PARAMETRIC_FORMS[fits["form_name"]][0]
        params.append(func(rs, *entry["popt"]))
    return np.array(params)


def plot_chi_interpolated(r, q, params_dict, rs, fits, ax=None, show_diff=True):
    """Compare χ(r) and χ(q) using original vs interpolated parameters.

    fits: output from fit_all_parameters()
    """
    from utils.fourier import chi_q_from_chi_r_fast
    from utils.utils_chi import chi00q, corradini_pz, get_gas_params

    kF, n0, NF = get_gas_params(rs)

    # χ with original parameters
    pi_orig = get_pi(q, rs)  # get_chi_interp(r, q, params_dict, rs)

    # χ with interpolated parameters
    params_interp = get_interpolated_params(rs, fits)
    # Create a temporary params_dict with interpolated values
    temp_dict = {rs: params_interp, "model": params_dict["model"]}
    pi_interp = get_pi_interp(r, q, temp_dict, rs)

    # Fourier transforms
    FT_q_orig, FT_chiq_orig = chi_q_from_chi_r_fast(r, pi_orig, qlist=None)
    FT_q_interp, FT_chiq_interp = chi_q_from_chi_r_fast(r, pi_interp, qlist=None)

    # Reference χ(q)
    fxc = corradini_pz(rs, q)
    vc = 4 * np.pi / q**2
    chi0q = chi00q(q, rs)
    chiq_ref = chi0q / (1 - chi0q * (vc + fxc))

    font_size = 10

    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: χ(r) comparison
    ax[0, 0].plot(
        kF * r[::10],
        pi_orig[::10] / (2 * kF**4 / pi),
        "k-",
        lw=1,
        label="Original params",
    )
    ax[0, 0].plot(
        kF * r[::30],
        pi_interp[::30] / (2 * kF**4 / pi),
        "r--",
        lw=1,
        label="Interpolated (mixed)"
        if fits.get("form_map")
        else f"Interpolated ({fits['form_name']})",
    )
    ax[0, 0].set_xlabel(r"$k_F r$", fontsize=font_size)
    ax[0, 0].set_ylabel(r"$\chi(r) / (2k_F^4/\pi)$", fontsize=font_size)
    ax[0, 0].set_xlim(0, 15)
    lim_upper = 0.0002  # max(chiR/(2*kF**4/pi**3))*.2
    ax[0, 0].set_ylim(-lim_upper / 1, lim_upper)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f"$r_s = {rs}$", fontsize=font_size)
    ax[0, 0].grid(True, alpha=0.3)

    # Top-right: χ(r) difference
    diff_r = (pi_interp - pi_orig) / (2 * kF**4 / pi)
    ax[0, 1].plot(kF * r, diff_r, "b-", lw=0.8)
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].set_xlabel(r"$k_F r$", fontsize=font_size)
    ax[0, 1].set_ylabel(r"$\Delta\chi(r) / (2k_F^4/\pi)$", fontsize=font_size)
    ax[0, 1].set_xlim(0, 15)
    ax[0, 1].set_title(f"Diff: max={np.max(np.abs(diff_r)):.2e}", fontsize=font_size)
    ax[0, 1].grid(True, alpha=0.3)

    # Bottom-left: χ(q) comparison
    ax[1, 0].plot(FT_q_orig / kF, -FT_chiq_orig / NF, "k-", lw=1, label="Original")
    ax[1, 0].plot(
        FT_q_interp / kF, -FT_chiq_interp / NF, "r--", lw=1, label="Interpolated"
    )
    ax[1, 0].plot(q / kF, -chiq_ref / NF, "g:", lw=1, alpha=0.7, label="Reference")
    ax[1, 0].set_xlabel(r"$q/k_F$", fontsize=font_size)
    ax[1, 0].set_ylabel(r"$-\chi(q)/N_F$", fontsize=font_size)
    ax[1, 0].set_xlim(0, 6)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(True, alpha=0.3)

    # Bottom-right: χ(q) difference
    diff_q = (FT_chiq_interp - FT_chiq_orig) / NF
    ax[1, 1].plot(FT_q_orig / kF, diff_q, "b-", lw=0.8)
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].set_xlabel(r"$q/k_F$", fontsize=font_size)
    ax[1, 1].set_ylabel(r"$\Delta\chi(q)/N_F$", fontsize=font_size)
    ax[1, 1].set_xlim(0, 6)
    ax[1, 1].set_title(f"Diff: max={np.max(np.abs(diff_q)):.2e}", fontsize=font_size)
    ax[1, 1].grid(True, alpha=0.3)

    _safe_tight_layout(fig)
    return {"max_diff_r": np.max(np.abs(diff_r)), "max_diff_q": np.max(np.abs(diff_q))}


def scan_chi_interpolation_error(params_dict, form_name="Pade[2/2]", rs_test=None):
    """Scan interpolation error in χ across multiple rs values.

    form_name : str or dict  (same as fit_all_parameters)
    Returns list of dicts with error metrics, and the fits object.
    """
    from input import q, r
    from utils.fourier import chi_q_from_chi_r_fast
    from utils.utils_chi import get_gas_params

    fits = fit_all_parameters(params_dict, form_name)

    rsl = sorted(
        [k for k in params_dict.keys() if isinstance(k, (float, int)) and k != "model"]
    )
    if rs_test is None:
        rs_test = rsl  # test all

    # Setup r, q grids

    results = []
    for rs in rs_test:
        kF, n0, NF = get_gas_params(rs)

        # Original χ
        pi_orig = get_pi(q, rs)  # get_chi_interp(r, q, params_dict, rs)
        FT_chiq_orig = get_piq(q, rs)

        # Interpolated χ
        params_interp = get_interpolated_params(rs, fits)
        temp_dict = {rs: params_interp, "model": params_dict["model"]}
        pi_interp = get_pi_interp(r, q, temp_dict, rs)
        FT_q_interp, FT_chiq_interp = chi_q_from_chi_r_fast(r, pi_interp, qlist=None)

        # Compute errors
        norm_r = 2 * kF**4 / pi
        norm_q = NF
        q_mask = q < 10.0 * kF  # focus on low-q region where χ(q) is most relevant
        diff_r = np.abs(pi_interp - pi_orig) / norm_r
        diff_q = np.abs(FT_chiq_interp - FT_chiq_orig) / norm_q

        # Relative to original signal
        rel_r = np.max(diff_r) / (np.max(np.abs(pi_orig)) / norm_r + 1e-30) * 100
        rel_q = (
            np.max(diff_q[q_mask])
            / (np.max(np.abs(FT_chiq_orig[q_mask])) / norm_q + 1e-30)
            * 100
        )

        rel_r = np.max(diff_r)
        rel_q = np.max(diff_q[q_mask] + 1e-30)

        MADE_e = np.sum(diff_r) / (np.sum(np.abs(pi_orig)) / norm_r + 1e-30)
        MADE_q = np.sum(diff_q[q_mask]) / (
            np.sum(np.abs(FT_chiq_orig[q_mask])) / norm_q + 1e-30
        )

        results.append(
            {
                "rs": rs,
                "max_diff_r": np.max(diff_r),
                "max_diff_q": np.max(diff_q),
                "rel_err_r_%": rel_r,
                "rel_err_q_%": rel_q,
                "MADE_r_%": MADE_e * 100,
                "MADE_q_%": MADE_q * 100,
            }
        )

    return results, fits
