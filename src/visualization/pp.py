import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from analysis.modes import get_constraints
from analysis.physics import get_chi, get_gas_params
from optimization.production import get_chi_interp
from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import chi00q, corradini_pz


def plot_parameters(params_dict):
    keys = list(params_dict.keys())
    rsl = [k for k in keys if isinstance(k, float)]
    # rsl = rsl[4:]
    rsl.sort()
    font_size = 6
    marker_size = 2
    lww = 0.7
    plt.rcParams["figure.dpi"] = 600
    # figsize=(5.92, 10.18)
    fig, ax = plt.subplots(nrows=4, ncols=1, constrained_layout=True, sharex=True)

    B0l = []
    B1l = []
    for j in range(2):
        coef1l = []
        coef2l = []

        for rs_i in range(len(rsl)):
            rs = rsl[rs_i]
            mode1 = params_dict[rs][0:2]
            mode2 = params_dict[rs][2:4]
            coef1, coef2 = mode1[j], mode2[j]
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

    C0, D0, C1, D1 = get_constraints(params_dict, rsl)

    ax[2].plot(rsl, C0, "k-o", label=r"$m=1$", lw=lww, markersize=marker_size)
    ax[2].plot(rsl, C1, "r-o", label=r"$m=2$", lw=lww, markersize=marker_size)
    # ax[3].set_xlabel(r'$r_s$',fontsize=font_size)
    ax[2].legend(fontsize=font_size)
    ax[2].set_ylabel(r"$C_m$", fontsize=font_size)
    ax[2].tick_params(axis="both", labelsize=font_size)

    ax[3].plot(rsl, D0, "k-o", label=r"$m=1$", lw=lww, markersize=marker_size)
    ax[3].plot(rsl, D1, "r-o", label=r"$m=2$", lw=lww, markersize=marker_size)
    # ax[3].set_xlabel(r'$r_s$',fontsize=font_size)
    ax[3].legend(fontsize=font_size)
    ax[3].set_ylabel(r"$D_m$", fontsize=font_size)
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
    chiR = get_chi(q, rs)
    chi_interpp = get_chi_interp(r, params_dict, rs)
    dr = r[1] - r[0]
    font_size = 8

    fig, ax = plt.subplots(nrows=2, ncols=1)

    fxc = corradini_pz(rs, q)
    vc = 4 * np.pi / q**2

    chi0q = chi00q(q, rs)
    chiq = chi0q / (1 - chi0q * (vc + fxc))

    FT_q, FT_chiq = chi_q_from_chi_r_fast(r, chi_interpp, qlist=None)

    if error:
        # ax[0].plot(kF*r[::10],chiR[::10]/NF,'k-',label=r'$\chi^h(r)$')
        ax[0].plot(
            kF * r[100::180],
            (chiR - chi_interpp)[100::180] / (2 * kF**4 / pi**3),
            "r-",
            markersize=1,
            label=r"$\chi_{M=2}^{h,\mathrm{interp.}}(r)$",
        )
        # ax[0].plot(kF*r[::180],np.abs((chiR-chi_interp)/chiR)[::180]*100,'ro',markersize=1,label=r'$\chi_{M=2}^{h,\mathrm{interp.}}(r)$')
        ax[0].plot(kF * r, 0 * kF * r, "k", lw=0.5)
        lim_upper = 0.0002  # max(chiR/(2*kF**4/pi**3))*.2
        ax[0].set_ylim(-lim_upper / 1, lim_upper)
        ax[0].set_xlim(0, 24)
        ax[0].set_xlabel(r"$k_F r$", fontsize=font_size, labelpad=2)
        ax[0].set_ylabel(r"$\Delta\chi(r)/6\pi n_0 N_\mathrm{F}$", fontsize=font_size)
        # ax[0].set_title(fr'$r_s = {rs}$',fontsize=font_size)

        # ax[1].plot(FT_q/kF,-np.abs(FT_chiq-chiq)/chiq*100,'r-.',label=r' $\chi^0(q)$ invFT',lw=1)
        ax[1].plot(FT_q / kF, (chiq - FT_chiq) / NF, "r-.", lw=1)

        # error_qmc = 0.2577 / 2
        limy = max(abs(chiq - FT_chiq) / NF) * 1.2  # error_qmc*2
        # plt.axhspan(-error_qmc,error_qmc, xmax=10,color='grey', alpha=0.3)
        ax[1].set_ylim(-limy, limy)
        # ax[1].set_ylim(0,10)
        ax[1].set_xlim(0, 10)
        ax[1].set_xlabel(r"$q/k_F$", fontsize=font_size, labelpad=2)
        ax[1].set_ylabel(r"$\Delta\chi(q)/N_\mathrm{F}$", fontsize=font_size)
        # ax[1].set_title(fr'$r_s = {rs}$')
        # ax[1].legend(fontsize=font_size)
    else:
        ax[0].plot(
            kF * r[::10], chiR[::10] / (2 * kF**4 / pi**3), "k-", label=r"$\chi^h(r)$"
        )
        ax[0].plot(
            kF * r[::180],
            chi_interpp[::180] / (2 * kF**4 / pi**3),
            "r-.",
            markersize=1,
            label=r"$\chi_{M=2}^{h,\mathrm{interp.}}(r)$",
        )
        ax[0].plot(kF * r, 0 * kF * r, "k", lw=0.5)
        lim_upper = max(chiR / (2 * kF**4 / pi**3)) * 1.2
        ax[0].set_ylim(-lim_upper / 1, lim_upper)
        ax[0].set_xlim(0, 12)
        ax[0].set_xlabel(r"$k_F r$", fontsize=font_size, labelpad=2)
        ax[0].set_ylabel(r"$\chi(r)/6\pi n_0 N_\mathrm{F}$", fontsize=font_size)
        # ax[0].set_title(fr'$r_s = {rs}$',fontsize=font_size)
        ax[0].legend(loc="lower right", fontsize=font_size)
        ax[0].tick_params(axis="both", labelsize=font_size)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, -1))  # force ×10^{-1}

        ax[0].yaxis.set_major_formatter(formatter)
        ax[0].ticklabel_format(axis="y", style="sci", scilimits=(-1, -1))

        ax[1].plot(q / kF, -chiq / NF, "k", label=r" $\chi(q)$ analytical", lw=1)
        ax[1].plot(FT_q / kF, -FT_chiq / NF, "r-.", label=r" $\chi(q)$ invFT", lw=1)
        ax[1].set_xlim(0, 10)
        ax[1].set_xlabel(r"$q/k_F$", fontsize=font_size, labelpad=2)
        ax[1].set_ylabel(r"$-\chi(q)/N_\mathrm{F}$", fontsize=font_size)
        # ax[1].set_title(fr'$r_s = {rs}$')
        ax[1].legend(fontsize=font_size)
        ax[1].tick_params(axis="both", labelsize=font_size)

    print(f"∫chi(r)r^2dr: {np.sum(chi_interpp * r**2) * dr:.6f}")
    # fig.subplots_adjust(hspace=1.4)
    fig.subplots_adjust(hspace=0.35, left=0.15, right=0.95, top=0.92, bottom=0.12)
