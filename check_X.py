# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from input import q, r
from optimization.fitting import guess_X
from optimization.models import delta_pi
from utils.fourier import chi_q_from_chi_r_fast
from utils.physics import get_B
from utils.utils_chi import (
    G_Moroni,
    chi00q,
    corradini_pz,
    get_chi,
    get_chi02,
    get_gas_params,
    get_pi,
)

plt.style.use(["science"])
plt.rcParams["figure.dpi"] = 300
rs = 0.01
kF, n0, NF = get_gas_params(rs)

chi0q = chi00q(q, rs)  # -chi00q(q,rs,interacting=False)[0]
vc = 4 * np.pi / q**2
fxc = corradini_pz(rs, q)
piq = chi0q / (1 - chi0q * fxc)
chiq = chi0q / (1 - chi0q * (vc + fxc))
chirpaq = chi0q / (1 - chi0q * vc)
piq2 = chiq / (1 + chiq * vc)

fig, ax = plt.subplots()

ax.plot(q / kF, -chiq / NF, "k", label=r" $\chi^0(q)$", lw=1)
ax.plot(q / kF, -chirpaq / NF, "k--", label=r" $\chi^{\mathrm{RPA}}(q)$", lw=1)
# ax.plot(q / kF,-piq / NF,"r-",label=r" $\Pi(q)$ ",lw=1)
# ax.plot(q / kF, -chi0q / NF, "k-.", label=r" $\chi^0(q)$ Lindhard", lw=1)
# ax.plot(q / kF, -piq2 / NF, "b-.", label=r" $\Pi_2(q)$", lw=1)
# ax.plot(q / kF, (1 - chi0q * fxc), "k-", label=r" $f_{xc}(q)$", lw=1)
# ax.plot(q / kF, 0 * q / kF, "k--", label=r" $f_{xc}(q) = 0$", lw=1)
ax.set_xlim(0, 4)

# %%
import matplotlib.pyplot as plt

from input import q
from utils.utils_chi import get_gas_params

plt.style.use(["science"])
plt.rcParams["figure.dpi"] = 300
rs = 5.2
kF, n0, NF = get_gas_params(rs)
factor = -6 * np.pi * n0 * NF

B = get_B(rs)
piR = get_pi(q, rs)
chiR = get_chi(q, rs)
chi0R = get_chi02(q, rs)

model = delta_pi
kFr0 = 0
kFr1 = 6

initial_guess = [
    1,
    2 * kF / 2.0 / np.pi,
    np.pi / 2 - 0.1,
    0.5,
    2 * kF / 2.0 / np.pi,
    np.pi / 2 + 0.1,
]

delta_chi_exact = -(chi0R - chiR) / factor
delta_pi_exact = -(chi0R - piR) / factor

"""
p_opt, p_cov = guess_X(
    r,
    rs,
    delta_chi_exact,
    model=model,
    initial_guess=initial_guess,
    kFr0=kFr0,
    kFr1=kFr1,
)

delta_chi_guess = model(r, rs, p_opt)
"""

fig, ax = plt.subplots()

ax.plot(kF * r[::10], -chi0R[::10] / factor, "k-.", label=r"$\chi_0(r)$")
ax.plot(kF * r[::10], -piR[::10] / factor, "r-", label=r"$\Pi(r)$")
ax.plot(kF * r[::10], -chiR[::10] / factor, "k-", label=r"$\chi(r)$")
# ax.plot(kF * r, delta_chi_guess, "r-", label=r"fit, $M=2$")

# plt.ylim(-10,20)
ax.plot(kF * r, 0 * kF * r, "k", lw=0.5)

lim_upper = max(-piR / factor)  # max(-delta_chi_exact / factor) * 3.03
ax.set_ylim(-lim_upper, lim_upper)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$k_F r$")
ax.set_title(rf"$r_s = {rs}$")

ax.legend()
plt.show()

scienceplots
# %%
# %%
import matplotlib.pyplot as plt
import numpy as np

from utils.utils_chi import chi00q, corradini_pz, get_chi

chi_reconstructed = chi0R - 6 * np.pi * n0 * NF * delta_chi_guess

fxc = corradini_pz(rs, q)
vc = 4 * np.pi / q**2

G = G_Moroni(rs, q)
fxc_Moroni = -vc * G

chi0q = chi00q(q, rs)
chiq = chi0q / (1 - chi0q * (vc + fxc))

FT_q, FT_chiq = chi_q_from_chi_r_fast(r, chi_reconstructed, qlist=q)

plt.plot(q / kF, -chiq / NF, "k", label=r" $\chi^0(q)$ analytical", lw=1)
plt.plot(
    FT_q / kF,
    -(FT_chiq - 0 * FT_chiq[0]) / NF,
    "r-.",
    label=r" $\chi^0(q)$ invFT",
    lw=1,
)

plt.xlim(0, 10)
plt.xlabel(r"$q/k_F$")
plt.ylabel(r"$-\chi^0(q)/n_0$")
# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from input import q, r
from optimization.fitting import exponential_cutoff_match
from optimization.models import X_r2_two_mode
from utils.physics import get_B, get_chi, get_chi0, get_gas_params

plt.style.use(["science"])
plt.rcParams["figure.dpi"] = 300

rs = 10
kF, n0, NF = get_gas_params(rs)
factor = -6 * np.pi * n0 * NF

B = get_B(rs)
chiR = get_chi(q, rs)
chi0R = get_chi0(r, rs)

model = X_r2_two_mode
gamma = 1
kFr0 = 0
kFr1 = 8

initial_guess = [
    1,
    2 * kF / (2 * np.pi),
    np.pi / 2,
    0.3,
    2 * kF / (2 * np.pi),
    np.pi / 2 - 1e-4,
]

X_exact = (
    chiR * (2 * kF * r) ** 4 / r
    + B * 2 * kF * np.cos(2 * kF * r)
    - B * np.sin(2 * kF * r) / r
) / r ** (gamma - 1)

p_opt, p_cov = guess_X(
    r,
    rs,
    X_exact,
    model=model,
    initial_guess=initial_guess,
    gamma=gamma,
    kFr0=kFr0,
    kFr1=kFr1,
)

X_guess = model(r, rs, p_opt, gamma)


r0 = 0.2

X_guess_cut = exponential_cutoff_match(X_guess, r, r0=r0)

fig, ax = plt.subplots()

ax.plot(kF * r[::10], X_exact[::10] / NF, "k-", label=r"$X_{\mathrm{exact}}(r)$")
ax.plot(kF * r, X_guess / NF, "r--", alpha=0.5, label=r"fit (raw)")
ax.plot(kF * r, X_guess_cut / NF, "b-", label=r"fit + cutoff")

ax.plot(kF * r, 0 * r, "k", lw=0.5)

lim_upper = max(X_exact / NF) * 0.03
ax.set_ylim(-lim_upper / 3, lim_upper)
ax.set_xlim(0, 1)
ax.set_xlabel(r"$k_F r$")
ax.set_title(rf"$r_s = {rs}$")

ax.legend()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import curve_fit

from input import q, r
from optimization.fitting import exponential_cutoff_match, guess_X
from optimization.models import X_r2_two_mode
from utils.physics import get_B
from utils.utils_chi import get_chi, get_chi0, get_chi02, get_gas_params

rs = 1
kF, n0, NF = get_gas_params(rs)
factor = -(6 * np.pi * n0 * NF)

B = get_B(rs)
chiR = get_chi(q, rs)
chi0R = get_chi02(q, rs)


fig, ax = plt.subplots()

fnn = chiR  # *(2 * kF * r) ** 4/r
fnn2 = chi0R  # *(2 * kF * r) ** 4/r
skip = 1

# ax.plot(kF * r, chiR / NF, "r--", alpha=0.5, label=r"fit (raw)")
ax.plot(
    kF * r[0::skip],
    (fnn2 - fnn)[0::skip] / (6 * np.pi * n0 * NF),
    "b-",
    label=r"fit + cutoff",
)
# ax.plot(
#    kF * r[0::skip], (fnn)[0::skip] / (6 * np.pi * n0 * NF), "r-", label=r"fit + cutoff"
# )

ax.plot(kF * r, 0 * r, "k", lw=0.5)

lim_upper = max(fnn / (6 * np.pi * n0 * NF)) * 4.3
ax.set_ylim(-lim_upper / 1, lim_upper)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$k_F r$")
ax.set_title(rf"$r_s = {rs}$")

ax.legend()

zerof = (fnn2 - fnn)[0] / (6 * np.pi * n0 * NF)
ax.scatter(0, zerof, s=100, c="r")


print(fnn[0], fnn2[0], zerof)


def model(r, A1, k1, phi1, alpha1, A2, k2, phi2, alpha2):
    y0 = (fnn2 - fnn)[0] / (6 * np.pi * n0 * NF)
    # A = y0 / np.cos(phi)
    # phi= np.arccos(y0/A )
    return A1 * np.cos(k1 * r + phi1) * np.exp(-alpha1 * r) + A2 * np.cos(
        k2 * r + phi2
    ) * np.exp(-alpha2 * r)


kFr0 = 0
kFr1 = 12
fit_idx0 = np.argmin(np.abs(kF * r - kFr0))
fit_idx1 = np.argmin(np.abs(kF * r - kFr1))
initial_guess = [1, 2 * kF, 0.1, 0.1, 0.5, 2 * kF, 0.2, 0.1]

p_opt, p_cov = curve_fit(
    model,
    r[fit_idx0:fit_idx1],
    (fnn2 - fnn)[fit_idx0:fit_idx1] / (6 * np.pi * n0 * NF),
    p0=initial_guess,
    maxfev=30000,
)
guess = model(r, *p_opt)
ax.plot(kF * r, guess, "g-.", label="fit to difference")
ax.legend()
plt.show()


# %%

chi_reconstructed = chi0R - 6 * np.pi * n0 * NF * guess
fig, ax = plt.subplots()
ax.plot(
    kF * r[::10],
    chiR[::10] / (6 * np.pi * n0 * NF),
    "k-",
    label=r"$\chi_{\mathrm{exact}}(r)$",
)
ax.plot(
    kF * r,
    chi_reconstructed / (6 * np.pi * n0 * NF),
    "r-.",
    label=r"reconstructed $\chi(r)$",
)
ax.plot(kF * r, 0 * r, "k", lw=0.5)

lim_upper = max(chiR / (6 * np.pi * n0 * NF)) * 4.3
ax.set_ylim(-lim_upper / 1, lim_upper)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$k_F r$")
ax.set_title(rf"$r_s = {rs}$")
# %%
import matplotlib.pyplot as plt
import numpy as np

from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import G_Moroni, chi00q, corradini_pz, get_chi

chi_reconstructed = chi0R - 6 * np.pi * n0 * NF * guess

fxc = corradini_pz(rs, q)
vc = 4 * np.pi / q**2

G = G_Moroni(rs, q)
fxc_Moroni = -vc * G

chi0q = chi00q(q, rs)
chiq = chi0q / (1 - chi0q * (vc + fxc))

FT_q, FT_chiq = chi_q_from_chi_r_fast(r, chi_reconstructed, qlist=q)

# Or directly at your desired r points (must lie within [r_dual.min, r_dual.max]):
# rlist = np.linspace(r_dual[0], 10.0, 400)
# chi_r = chi_r_from_chi_q_fast(qlist, chi_q, rlist=rlist)


plt.plot(q / kF, -chiq / NF, "k", label=r" $\chi^0(q)$ analytical", lw=1)
plt.plot(
    FT_q / kF,
    -(FT_chiq - 0 * FT_chiq[0]) / NF,
    "r-.",
    label=r" $\chi^0(q)$ invFT",
    lw=1,
)

plt.xlim(0, 10)
plt.xlabel(r"$q/k_F$")
plt.ylabel(r"$-\chi^0(q)/n_0$")
# %%

# %%
