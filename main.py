import time

import numpy as np
from scipy.optimize import curve_fit

from analysis.physics import ElectronGas, get_gas_params
from optimization.fitting import guess_AB
from optimization.models import model3, model4

# import scienceplots


# plt.style.use(['science'])
# plt.rcParams['figure.dpi'] = 300

rs = 1
mygas = ElectronGas(rs)
kF, n0, NF = get_gas_params(rs)


chi0R = mygas.chi0R
r = mygas.r
q = mygas.q
chiR = mygas.chiR


B_exact = chiR * (2 * kF * r) ** 4 / r
B, pcov = guess_AB(r, kF, B_exact, model3, initial_guess=None, kFr0=24, kFr1=39)

if pcov[0][0] < 1e-6:
    A_exact = chiR * (2 * kF * r) ** 4 / r + B * 2 * kF * np.cos(2 * kF * r)
    A, pcov = guess_AB(r, kF, A_exact, model4, initial_guess=None, kFr0=24, kFr1=39)

    print("Calculated A: ", A, pcov)
    if pcov[0][0] < 1e-6:
        print("Found reasonable A and B. Proceeding...")
    else:
        raise RuntimeError("Failed to find reasonable A (pcov too large)")

else:
    raise RuntimeError("Failed to find reasonable B (pcov too large)")

X_exact = (
    chiR * (2 * kF * r) ** 4 / r
    + B * 2 * kF * np.cos(2 * kF * r)
    - A * np.sin(2 * kF * r) / r
)
Y0 = (B - A) * 2 * kF


def model5(t, *params):
    # params = [A1, gamma1, f1, phi1, A2, gamma2, f2, phi2, ...]
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    M = len(params) // 4
    # print(len(params))
    for m in range(M):
        if m == M:
            g = params[4 * m + 1]
            f = params[4 * m + 2]
            phi = params[4 * m + 3]
            Aa = (Y0 - y[0]) / np.cos(phi)
            y += Aa * np.exp(-abs(g) * t) * np.sin(2 * np.pi * f * t + phi)
        else:
            Aa = params[4 * m + 0]
            g = params[4 * m + 1]
            f = params[4 * m + 2]
            phi = params[4 * m + 3]
            y += Aa * np.exp(-abs(g) * t) * np.sin(2 * np.pi * f * t + phi) * t
    return y


modell = model5

fit_idx = np.argmin(np.abs(kF * r - 8))
# t, y as before
rep = [1, 1, 2 * kF / (2 * np.pi), 0.3]
n = 2
initial_guess = n * rep  # need reasonable guesses for A, gamma, f, phi

print(f"Fitting X with {modell}...")
start_time = time.time()
p_opt, p_cov = curve_fit(
    modell, r[0:fit_idx], X_exact[0:fit_idx], p0=initial_guess, maxfev=30000
)
end_time = time.time()
X = modell(r, *p_opt)
# how to get the name of f{modell}?
print(f"Fitting completed in {end_time - start_time:.2f} seconds.")
print("Optimized parameters:", p_opt)
print("Parameters saved to 'fitted_params.npy'")
np.save("fitted_params.npy", p_opt)
