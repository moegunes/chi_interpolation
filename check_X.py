import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from analysis.physics import get_B, get_chi, get_gas_params
from input import q, r
from optimization.fitting import guess_X
from optimization.models import X_r2_two_mode_nl

plt.style.use(["science"])
plt.rcParams["figure.dpi"] = 300


rs = 2
kF, n0, NF = get_gas_params(rs)
factor = -6 * np.pi * n0 * NF

B = get_B(rs)
chiR = get_chi(q, rs)

model = X_r2_two_mode_nl
gamma = 1
kFr0 = 0
kFr1 = 12

initial_guess = [
    0.3,
    1 * kF / (2 * np.pi),
    1,
    2 * kF / (2 * np.pi),
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


fig, ax = plt.subplots()

ax.plot(kF * r[::10], X_exact[::10] / NF, "k-", label=r"$X(r)$")
ax.plot(kF * r, X_guess / NF, "r-", label=r"fit, $M=2$")

# plt.ylim(-10,20)
ax.plot(kF * r, 0 * kF * r, "k", lw=0.5)

lim_upper = max(X_exact / NF) * 1.03
ax.set_ylim(-lim_upper / 3, lim_upper)
ax.set_xlim(0, 70)
ax.set_xlabel(r"$k_F r$")
ax.set_title(rf"$r_s = {rs}$")

ax.legend()
plt.show()

scienceplots
