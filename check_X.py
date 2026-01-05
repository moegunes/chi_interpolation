# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from input import q, r
from optimization.fitting import guess_X
from optimization.models import delta_chi_nl
from utils.physics import get_B
from utils.utils_chi import get_chi, get_chi02, get_gas_params

plt.style.use(["science"])
plt.rcParams["figure.dpi"] = 300
rs = 10
kF, n0, NF = get_gas_params(rs)
factor = -6 * np.pi * n0 * NF

B = get_B(rs)
chiR = get_chi(q, rs)
chi0R = get_chi02(q, rs)

model = delta_chi_nl
kFr0 = 0
kFr1 = 12

initial_guess = [
    0.5,
    1 * kF / 2.0 / np.pi,
    0.3,
    1 * kF / 2.0 / np.pi,
]

delta_chi_exact = -(chi0R - chiR) / factor


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


fig, ax = plt.subplots()

ax.plot(kF * r[::10], delta_chi_exact[::10], "k-", label=r"$X(r)$")
ax.plot(kF * r, delta_chi_guess, "r-", label=r"fit, $M=2$")

# plt.ylim(-10,20)
ax.plot(kF * r, 0 * kF * r, "k", lw=0.5)

lim_upper = 0.08  # max(-delta_chi_exact / factor) * 3.03
ax.set_ylim(-lim_upper, lim_upper)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$k_F r$")
ax.set_title(rf"$r_s = {rs}$")

ax.legend()
plt.show()

scienceplots

# %%
import matplotlib.pyplot as plt
import numpy as np

from utils.fourier import chi_q_from_chi_r_fast
from utils.utils_chi import G_Moroni, chi00q, corradini_pz, get_chi

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
