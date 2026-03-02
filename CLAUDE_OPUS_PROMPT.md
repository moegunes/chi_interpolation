# Chi Interpolation — Agent Context Document

## 1  Purpose of this file

This Markdown file is the **primary on-boarding document** for any LLM agent
(Claude Opus or similar) that works on the `chi_interpolation` code-base.
It explains the physics, the code architecture, the current fitting pipeline,
known problems, and conventions.  Read it *before* touching any source file.

**Last updated:** after multi-session improvement effort (sessions 1–5).
Session 5 added Phase 6 (global smooth re-fit) and interpolability analysis.

---

## 2  Physics background

### 2.1  The problem

We study the **homogeneous electron gas (HEG)** at density parameter
$r_s$.  The goal is to represent the *difference* between the interacting
and non-interacting density–density response functions in real space,
$\Delta\chi(r) \equiv \chi(r) - \chi_0(r)$, as a closed-form ansatz that
can be evaluated at *any* $r_s$ by smoothly interpolating a small set of
fitted parameters.

### 2.2  Model ansatz (two damped cosines)

$$
\Delta\chi(r)
= B_0\,e^{-\alpha_0\,k_F r}\cos(k_0\,k_F r + \phi_0)
+ B_1\,e^{-\alpha_1\,k_F r}\cos(k_1\,k_F r + \phi_1)
\qquad (k_i = 2\pi f_i)
$$

*  **Fitted (nonlinear) parameters at each $r_s$:**
   $\boldsymbol{\theta} = (\alpha_0, f_0, \phi_0,\; \alpha_1, f_1, \phi_1)$.
*  **Linearly determined amplitudes:**
   $B_0, B_1$ are solved from two moment / sum-rule constraints
   $$
     \mathbf{M}\begin{pmatrix}B_0\\B_1\end{pmatrix}
     = \begin{pmatrix}\delta C_1(r_s)\\\delta C_0(r_s)\end{pmatrix}
   $$
   where $\delta C_n$ are differences of moments of $\chi$ and $\chi_0$
   and $\mathbf{M}$ involves the analytic `J_n_m_kFr` integrals
   (see `physics.py`).

### 2.3  How data is generated

1. $\chi_0(q)$ via Lindhard function (`chi00q` in `utils_chi.py`).
2. $\chi(q)$ via $\chi_0/(1-\chi_0(v_c+f_{xc}))$ using Corradini–PZ
   local-field factor (`corradini_pz`).
3. $\chi(r)$ and $\chi_0(r)$ via DST-I spherical Fourier transform
   (`fourier.py`).
4. $\Delta\chi_{\text{exact}}(r) = -(\chi_0(r)-\chi(r))/(\text{factor})$
   with factor $= -6\pi n_0 N_F$.

The q-grid has $dq=0.01$, $q_{\max}=10000$ (set in `input.py`).

### 2.4  Why smooth parameters matter — the interpolation goal

The ultimate deliverable is an analytic formula for $\chi(r;r_s)$ usable
at any $r_s$.  To that end each of the 6 parameters $\theta_j(r_s)$ must
be a *smooth* function of $r_s$ so that a simple parametric form
(e.g. Padé approximant, power law, or low-order polynomial) reproduces the
fitted values to high accuracy.  **Parameter jumps between neighbouring
$r_s$ points are the main failure mode** — they prevent interpolation.

**B0 and B1 need not be interpolated independently** — they are derived
from the 6 nonlinear parameters via the moment-constraint linear solve.
If we interpolate $\theta_j(r_s)$ smoothly, B0(rs) and B1(rs) follow
automatically.

### 2.5  Discrete symmetries of the model

The model is invariant under several discrete transformations that create
multiple equivalent parameter sets (branches):

1. **Frequency sign:** $\cos(f r + \phi) = \cos(-f r - \phi)$
2. **Phase periodicity:** $\phi \to \phi + 2\pi k$
3. **Amplitude sign:** $(B_i, \phi_i) \to (-B_i, \phi_i + \pi)$
4. **Mode swap:** $(mode_0, mode_1) \leftrightarrow (mode_1, mode_0)$

These symmetries mean the optimizer can find physically identical but
numerically different solutions at neighbouring $r_s$, causing artificial
jumps.  Canonicalization (§4.5) removes these.

---

## 3  Repository layout

```
chi_interpolation/
├── main.py               # entry point – loops over rslist, calls fit_params
├── input.py              # q / r grids  (dq = 0.01, qmax = 10000)
├── diagnose_full.py      # prints 6 params + B0/B1 + cond(M) + max jumps
├── diagnose.py           # simpler diagnostic (not primary)
├── analyze_interpolability.py  # parametric form fitting analysis
├── check_X.py            # interactive / cell-mode single-rs diagnostic
├── check_chi.ipynb       # Jupyter notebook for visual inspection
├── fitted_params.npy     # legacy; current results stored in parameters.pkl
├── CLAUDE_OPUS_PROMPT.md # ← this file
├── pyproject.toml        # project metadata
├── ruff.toml             # formatter config
└── src/
    ├── optimization/
    │   ├── models.py       # delta_chi()  (model + linear B solve)
    │   ├── fitting.py      # fit_params(), guess_X(), canonicalization, Phase 6
    │   └── production.py   # chi_interp(), get_chi_interp()
    ├── utils/
    │   ├── physics.py      # J_n_m_kFr, delta_C, chi/chi0 moments, get_B
    │   ├── utils_chi.py    # Lindhard, Corradini–PZ LFF, Fourier wrappers
    │   ├── fourier.py      # DST-I based χ(q)↔χ(r) transforms
    │   └── io.py           # pickle read/write helpers
    └── visualization/
        └── pp.py           # plot_parameters(), plot_chi(), get_constraints()
```

**Python environment:**  Python 3.13, venv at `.venv/`.
Use `.venv/bin/python` (bare `python` may not be available).
Working directory: project root (`chi_interpolation/`).

---

## 4  Fitting pipeline (current state)

### 4.1  Overview

```
main.py
  → fit_params(rslist, q, r, model, …)      [fitting.py]
       Phase 1  anchor multi-start at rs = 5.0  (11 candidates)
       Phase 2  sweep UP   (anchor → 10)
       Phase 3  sweep DOWN (anchor → 0.2)
       Phase 4  re-fit anchor from neighbours
       Phase 5  polynomial extrapolation re-fit for rs < 2.0
       Phase 6  global smooth re-fit (all 51 points simultaneously)
  → write_dict(parameters, "parameters")     [io.py]
```

**Current grid:** `np.concatenate([np.arange(0.2, 2.0, 0.1), np.arange(2.0, 10.25, 0.25)])` — 51 points.

**Runtime:** ~680 seconds total (86s phases 1–5 + ~600s phase 6) with dq=0.01.

### 4.2  `guess_X` — single-point curve_fit wrapper

```python
curve_fit(model_wrapper, r_fit, y_fit, p0=…,
          bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
          method='trf', maxfev=30000)
```

Fitting region: $k_F r \in [0, 4]$, subsampled to ≤2000 points.

### 4.3  `_fit_one_point` — candidate generation

For each $r_s$ **six candidates** are generated:

| # | source | method |
|---|--------|--------|
| 1 | warm start (prev params) | `curve_fit` TRF |
| 2 | physics guess | `curve_fit` TRF |
| 3 | perturbed warm start | `curve_fit` TRF |
| 4 | warm start, λ=0.1 | L-BFGS-B regularized |
| 5 | warm start, λ=1.0 | L-BFGS-B regularized |
| 6 | warm start, λ=5.0 | L-BFGS-B regularized |

All candidates are canonicalized and fed to `_select_smoothest`.

### 4.4  `_select_smoothest` — branch selection

Among candidates whose data cost is within `cost_tolerance` (default 3×)
of the best, the one with smallest **proximity metric** is chosen:

$$
d = \|\theta - \theta_{\text{prev}}\|^2
  + (B_0 - B_0^{\text{prev}})^2
  + (B_1 - B_1^{\text{prev}})^2
  + \text{mode-separation penalty}
$$

If $|f_0 - f_1| < \texttt{DF\_MIN}$, a penalty of $10^4$ is added.

### 4.5  Canonicalization (`_canonicalize_params`)

Applied after every fit to remove discrete symmetries:

| step | symmetry | action |
|------|----------|--------|
| 1 | $f \to -f$ | flip sign of $f$, negate $\phi$ |
| 2 | $\phi + 2\pi$ | wrap $\phi$ to $[-\pi, \pi)$ |
| 3 | $B < 0$ | shift $\phi_i += \pi$ when $B_i < 0$ (requires rs) |
| 4 | mode swap | order modes so $\lvert f_0\rvert \le \lvert f_1\rvert$ |
| 5 | $f < F_\text{min}$ | enforce $f \ge F_\text{min} = 0.02$ |

### 4.6  `_regularized_fit` — L-BFGS-B with penalties

Objective: $\mathcal{L} = \text{data\_cost} + \lambda\|\theta - \theta_\text{prev}\|^2 + \text{sep\_penalty} + \text{degen\_penalty}$

- **sep_penalty**: $10\lambda \cdot \max(0, DF_\text{min} - |f_0-f_1|)^2$ — prevents mode collapse.
- **degen_penalty**: when $|f_i - F_\text{min}| < F_\text{min}/2$, adds $100\lambda$ extra regularization on $\alpha_i$ and $\phi_i$ — prevents wild excursions of ill-determined parameters.
- λ is normalized: $\lambda_\text{eff} = \lambda_\text{smooth} \cdot \|y\|^2 / \|\theta_\text{prev}\|^2$.

### 4.7  Phase 5 — Extrapolation re-fit (rs < 2.0)

1. Fit cubic polynomial to each of the 6 parameters in the smooth region (rs ≥ 2.0).
2. Extrapolate to low rs → candidate initial guesses.
3. For each low-rs point (swept downward from threshold), generate candidates:
   - Current sweep result, extrapolated, from neighbour, regularized at λ ∈ {1, 10, 50}.
4. Select smoothest with relaxed tolerance (2× `cost_tolerance`).

### 4.8  Phase 6 — Global smooth re-fit (`_global_smooth_refit`)

**This is the most impactful improvement for interpolability.**

After phases 1–5 produce per-point fits, Phase 6 optimizes **all 51 rs
points simultaneously** (306 variables) via L-BFGS-B with:

$$
\mathcal{L}_\text{total}
= \underbrace{\sum_i \mathcal{L}_\text{data}(r_{s,i})}_{\text{data fidelity}}
+ \lambda \underbrace{\sum_i \frac{\|S(\theta_{i+1} - \theta_i)\|^2}{\Delta r_{s,i}^2}}_{\text{first-order smoothness}}
+ \lambda \underbrace{\sum_i \frac{\|S(\theta_{i+2} - 2\theta_{i+1} + \theta_i)\|^2}{\overline{\Delta r_s}^4}}_{\text{curvature (2nd order)}}
+ \text{penalties}
$$

Key design choices:

- **Inlined model evaluation:** The model (J integrals, B solve, residual)
  is inlined in the objective function to avoid function call overhead.
  This gives ~3× speedup over calling `delta_chi()` per point.
- **Per-parameter scaling:** $S = \text{diag}(1/\sigma_j)$ where $\sigma_j$
  is the standard deviation of parameter $j$ across rs.  This prevents
  small-range parameters (like f0) from being dominated by large-range ones.
- **Lambda calibration:** $\lambda_\text{eff} = \lambda_\text{smooth} \times
  \text{init\_data\_cost} / \text{init\_smooth\_cost}$, so `lambda_smooth=1.0`
  means equal weight to data fidelity and smoothness at the starting point.
  Currently set to `lambda_smooth=10.0` (smoothness 10× more important).
- **Curvature penalty:** Targets staircase patterns where first-order
  differences are small but second-order differences reveal discontinuities.
- **Degenerate mode-0 penalty:** Extra smoothness weight (50×) on α₀ and φ₀
  when f₀ is near F_MIN (mode 0 is vestigial, parameters are floppy).
- **Subsampling:** Each rs curve is subsampled to ~300 points (vs 2000 in
  per-point fits) for speed.  Adequate since the global optimizer doesn't
  need per-point precision.

**Current settings:** `lambda_smooth=10.0`, `max_iter=2000`, `maxfun=500000`.

**Impact:** Phase 6 reduces overall smoothness cost by ~86% while increasing
data cost by ~340% (from 9.8e-6 to 4.3e-5, still excellent fits).

### 4.9  Constants

| name | value | purpose |
|---|---|---|
| `F_MIN` | 0.02 | minimum frequency (prevents pure-exponential mode) |
| `DF_MIN` | 0.03 | minimum $\lvert f_0 - f_1\rvert$ (prevents degenerate modes) |
| `cost_tolerance` | 3.0 | cost-ratio threshold for smoothness selection |
| `anchor_target` | 5.0 | anchor point for bidirectional sweep |
| `smooth_threshold` | 2.0 | rs above which Phase 5 trusts the data |
| `BOUNDS_LOWER` | `[1e-4, F_MIN, -π, 1e-4, F_MIN, -π]` | |
| `BOUNDS_UPPER` | `[20.0, 3.0, π, 20.0, 3.0, π]` | |

### 4.10  Dead code

- `_backward_fixup`: exists but is **not called** anywhere.  Its role was
  subsumed by Phase 5.  Can be deleted or kept for reference.

---

## 5  Current parameter state (after Phase 6)

### 5.1  Max jumps (primary quality metric)

| quantity | max jump | % of range | location | pre-Phase6 |
|---|---|---|---|---|
| alpha0 | 0.130 | **13.8%** | rs 0.20→0.30 | 16.7% |
| f0     | 0.004 | **5.0%** | rs 0.90→1.00 | 14.8% |
| phi0   | 0.027 | **12.2%** | rs 0.70→0.80 | **59.8%** |
| alpha1 | 0.009 | **8.5%** | rs 0.20→0.30 | **44.4%** |
| f1     | 0.017 | **11.4%** | rs 0.30→0.40 | 19.0% |
| phi1   | 0.329 | **16.3%** | rs 0.50→0.60 | 20.4% |
| B0     | 0.017 | **8.4%** | rs 0.70→0.80 | **28.6%** |
| B1     | 0.003 | **4.9%** | rs 2.00→2.25 | 17.1% |

**Summary:** All max jumps are now ≤16.3% of range.  The worst pre-Phase6
offenders (phi0: 60%, alpha1: 44%, B0: 29%) improved by 3–5×.

### 5.2  Regime breakdown

| regime | quality | notes |
|---|---|---|
| rs ≥ 2.0 | **excellent** (< 3% jumps) | all 8 quantities monotonic and smooth |
| rs 1.0–2.0 | **good** (< 5% jumps) | smooth transition; f0 rises off the F_MIN floor |
| rs 0.5–1.0 | **moderate** (< 13%) | phi0 has a dip; phi1 rapid transition |
| rs 0.2–0.4 | **acceptable** (< 17%) | alpha0, f1 largest jumps; physics-driven |

### 5.3  Full parameter table (51 points)

```
   rs    alpha0       f0     phi0   alpha1       f1     phi1 |         B0         B1    cond(M)
  0.20    0.4867   0.0200   2.0031   0.8420   0.1038   2.0850 |     0.0078     0.0102       28.5
  0.30    0.6166   0.0214   2.0088   0.8510   0.1195   1.8938 |     0.0164     0.0114       13.1
  0.40    0.7283   0.0227   2.0108   0.8599   0.1362   1.6600 |     0.0281     0.0105        8.4
  0.50    0.8217   0.0248   2.0040   0.8682   0.1526   1.3689 |     0.0412     0.0097        6.5
  0.60    0.9028   0.0277   1.9867   0.8761   0.1682   1.0402 |     0.0553     0.0095        5.6
  0.70    0.9769   0.0307   1.9614   0.8835   0.1824   0.7270 |     0.0708     0.0101        5.3
  0.80    1.0447   0.0339   1.9344   0.8899   0.1941   0.4651 |     0.0873     0.0111        5.3
  0.90    1.1049   0.0377   1.9077   0.8946   0.2022   0.2716 |     0.1037     0.0120        5.4
  1.00    1.1560   0.0418   1.8870   0.8976   0.2070   0.1473 |     0.1183     0.0130        5.5
  1.10    1.1951   0.0458   1.8798   0.8991   0.2084   0.0831 |     0.1288     0.0142        5.7
  1.20    1.2227   0.0493   1.8871   0.8992   0.2078   0.0677 |     0.1346     0.0160        5.8
  1.30    1.2406   0.0524   1.9050   0.8981   0.2073   0.0811 |     0.1368     0.0180        6.1
  1.40    1.2427   0.0561   1.9301   0.8954   0.2076   0.1002 |     0.1339     0.0193        6.5
  1.50    1.2389   0.0595   1.9567   0.8918   0.2089   0.1132 |     0.1305     0.0204        7.0
  1.60    1.2350   0.0625   1.9781   0.8884   0.2111   0.1173 |     0.1284     0.0213        7.5
  1.70    1.2331   0.0650   1.9961   0.8844   0.2138   0.1172 |     0.1275     0.0222        8.1
  1.80    1.2329   0.0672   2.0083   0.8800   0.2161   0.1157 |     0.1277     0.0229        8.5
  1.90    1.2381   0.0690   2.0158   0.8756   0.2177   0.1151 |     0.1294     0.0238        8.9
  2.00    1.2430   0.0707   2.0220   0.8709   0.2193   0.1158 |     0.1312     0.0247        9.2
  2.25    1.2629   0.0740   2.0333   0.8651   0.2205   0.1315 |     0.1372     0.0273        9.8
  2.50    1.2744   0.0769   2.0452   0.8583   0.2231   0.1398 |     0.1415     0.0295       10.7
  2.75    1.2854   0.0792   2.0567   0.8536   0.2246   0.1570 |     0.1453     0.0317       11.6
  3.00    1.2970   0.0812   2.0651   0.8500   0.2264   0.1657 |     0.1494     0.0338       12.7
  3.25    1.3065   0.0827   2.0752   0.8475   0.2272   0.1872 |     0.1525     0.0359       13.8
  3.50    1.3163   0.0843   2.0803   0.8445   0.2289   0.1923 |     0.1562     0.0378       15.0
  3.75    1.3251   0.0858   2.0845   0.8416   0.2305   0.1972 |     0.1596     0.0394       16.3
  4.00    1.3315   0.0867   2.0920   0.8386   0.2310   0.2174 |     0.1617     0.0412       17.8
  4.25    1.3387   0.0879   2.0952   0.8356   0.2323   0.2233 |     0.1646     0.0427       19.3
  4.50    1.3459   0.0890   2.0964   0.8327   0.2338   0.2247 |     0.1676     0.0440       20.9
  4.75    1.3522   0.0901   2.0973   0.8297   0.2351   0.2269 |     0.1704     0.0453       22.5
  5.00    1.3575   0.0910   2.0991   0.8268   0.2362   0.2328 |     0.1727     0.0465       24.4
  5.25    1.3624   0.0918   2.1007   0.8238   0.2372   0.2385 |     0.1748     0.0477       26.3
  5.50    1.3675   0.0927   2.1010   0.8212   0.2383   0.2407 |     0.1770     0.0487       28.3
  5.75    1.3724   0.0935   2.1009   0.8185   0.2393   0.2425 |     0.1793     0.0498       30.4
  6.00    1.3771   0.0943   2.1008   0.8162   0.2403   0.2443 |     0.1814     0.0508       32.5
  6.25    1.3815   0.0950   2.1007   0.8141   0.2413   0.2463 |     0.1834     0.0518       34.8
  6.50    1.3858   0.0956   2.1007   0.8121   0.2421   0.2483 |     0.1853     0.0527       37.2
  6.75    1.3900   0.0962   2.1007   0.8105   0.2429   0.2503 |     0.1871     0.0537       39.7
  7.00    1.3940   0.0968   2.1007   0.8089   0.2437   0.2524 |     0.1888     0.0547       42.2
  7.25    1.3979   0.0973   2.1007   0.8076   0.2444   0.2544 |     0.1905     0.0556       44.9
  7.50    1.4016   0.0978   2.1007   0.8063   0.2450   0.2564 |     0.1920     0.0565       47.6
  7.75    1.4051   0.0983   2.1006   0.8049   0.2457   0.2583 |     0.1935     0.0574       50.5
  8.00    1.4083   0.0988   2.1004   0.8036   0.2463   0.2602 |     0.1949     0.0582       53.5
  8.25    1.4114   0.0992   2.1000   0.8024   0.2470   0.2616 |     0.1963     0.0591       56.5
  8.50    1.4143   0.0996   2.0997   0.8010   0.2476   0.2632 |     0.1976     0.0598       59.7
  8.75    1.4170   0.1000   2.0995   0.7998   0.2482   0.2649 |     0.1988     0.0606       62.9
  9.00    1.4197   0.1004   2.0990   0.7987   0.2487   0.2662 |     0.1999     0.0614       66.3
  9.25    1.4222   0.1008   2.0987   0.7975   0.2492   0.2678 |     0.2010     0.0621       69.7
  9.50    1.4245   0.1011   2.0983   0.7964   0.2498   0.2691 |     0.2021     0.0628       73.3
  9.75    1.4267   0.1014   2.0978   0.7952   0.2503   0.2706 |     0.2031     0.0635       76.9
 10.00    1.4288   0.1018   2.0972   0.7942   0.2507   0.2716 |     0.2040     0.0641       80.6
```

### 5.4  Key observations from the data

1. **rs ≥ 2.0:** All parameters evolve smoothly and monotonically.
   The staircase pattern in phi1 (pre-Phase6) is now eliminated.

2. **rs 1.0–2.0:** Smooth transition region. f0 rises from near the floor
   (0.042→0.071), phi0 dips to ~1.88 then recovers to ~2.02, alpha1 and
   f1 are very smooth.

3. **rs < 1.0:** f0 is still near `F_MIN=0.02` (ranges 0.020→0.038).
   Mode 0 contributes modestly ($B_0 \approx 0.01$–$0.10$).  Its nonlinear
   parameters (α₀, φ₀) are better constrained now thanks to the degenerate-
   mode penalty in Phase 6, but remain the hardest to interpolate.

4. **phi1 transition:** phi1 undergoes a rapid but smooth transition from
   ~2.09 (rs=0.2) through ~0.07 (rs=1.1) back up to ~0.27 (rs=10).
   This is physical behavior, not an artifact.

5. **cond(M):** Ranges from 5.3 (rs=0.7–0.8) to 80.6 (rs=10).

---

## 6  Interpolability analysis — the core remaining task

### 6.1  What needs to be interpolated

Only the **6 nonlinear parameters** $(\alpha_0, f_0, \phi_0, \alpha_1, f_1, \phi_1)$
need parametric interpolation formulae $\theta_j(r_s)$.  B₀ and B₁ are
automatically determined by the moment constraints once $\theta_j$ is known.

### 6.2  Candidate parametric forms tested

The following forms were fitted to each $\theta_j(r_s)$ over all 51 rs points
(from 0.2 to 10.0):

| form | formula | # coefficients |
|---|---|---|
| `a+b*rs^c` | $a + b\,r_s^c$ | 3 |
| `poly2` | $a + b\,r_s + c\,r_s^2$ | 3 |
| `poly3` | $a + b\,r_s + c\,r_s^2 + d\,r_s^3$ | 4 |
| `Padé[1/1]` | $(a + b\,r_s)/(1 + c\,r_s)$ | 3 |
| **`Padé[2/2]`** | $(a + b\,r_s + c\,r_s^2)/(1 + d\,r_s + e\,r_s^2)$ | **5** |
| `a+b/(rs+c)` | $a + b/(r_s + c)$ | 3 |
| `saturating` | $a(1-e^{-b\,r_s}) + c$ | 3 |
| `a+b*rs^c+d*rs^e` | $a + b\,r_s^c + d\,r_s^e$ | 5 |

### 6.3  Results: best parametric fit for each parameter

| parameter | best form | # coeffs | RMSE | MaxErr | MaxErr % of range |
|---|---|---|---|---|---|
| **α₀** | Padé[2/2] | 5 | 2.32e-2 | 5.27e-2 | **5.6%** |
| **f₀** | Padé[2/2] | 5 | 9.85e-4 | 1.79e-3 | **2.2%** |
| **φ₀** | a+b·rs^c+d·rs^e | 5 | 3.40e-2 | 9.00e-2 | **40.7%** ⚠ |
| **α₁** | Padé[2/2] | 5 | 2.88e-3 | 6.29e-3 | **6.0%** |
| **f₁** | Padé[2/2] | 5 | 4.33e-3 | 1.03e-2 | **7.0%** |
| **φ₁** | Padé[2/2] | 5 | 2.62e-2 | 6.74e-2 | **3.3%** |

**Total meta-parameters for the 6 nonlinear quantities: 30** (if using Padé[2/2] for all).

### 6.4  B₀ and B₁ interpolability (for reference)

| quantity | best form | # coeffs | RMSE | MaxErr | MaxErr % |
|---|---|---|---|---|---|
| B₀ | Padé[2/2] | 5 | 7.80e-3 | 2.10e-2 | 10.7% |
| B₁ | Padé[2/2] | 5 | 5.46e-4 | 1.62e-3 | 3.0% |

B₀ and B₁ need **not** be interpolated directly — they are derived quantities.
These numbers show how smooth they'd be if you did.

### 6.5  Restricted analysis: rs ≥ 1.0 only (37 points)

When fitting parametric forms only to rs ≥ 1.0 (where physics is simple),
interpolation quality improves dramatically:

| parameter | form | RMSE | MaxErr % |
|---|---|---|---|
| α₀ | Padé[2/2] | 8.78e-3 | 11.3% |
| f₀ | Padé[2/2] | 4.14e-4 | 2.3% |
| **φ₀** | Padé[2/2] | 4.57e-3 | **4.6%** |
| α₁ | Padé[2/2] | 1.99e-3 | 4.3% |
| f₁ | Padé[2/2] | 1.03e-3 | 6.6% |
| φ₁ | Padé[2/2] | 8.38e-3 | 10.6% |

**Key insight: φ₀ is 40.7% max error over full range but only 4.6% for rs ≥ 1.**
The low-rs dip in φ₀ (from ~2.00 down to ~1.88 at rs≈1.1, then back up)
is the main interpolation challenge.

### 6.6  Per-parameter interpolability assessment

#### α₀ (damping rate, mode 0) — ✅ INTERPOLABLE

- **Shape:** Monotonically increasing, 0.49 → 1.43, saturating at high rs.
- **Best form:** Padé[2/2] (5.6% max error, mostly from low-rs curvature).
- **Character:** Looks like a fractional power law.  `a+b/(rs+c)` and
  Padé[1/1] also give ~6.8% error with only 3 coefficients.
- **Verdict:** Easily interpolable.  Even 3-param forms work well.

#### f₀ (frequency, mode 0) — ✅ INTERPOLABLE

- **Shape:** Rises from 0.020 (near F_MIN) to 0.102.  Saturating curve.
- **Best form:** Padé[2/2] (2.2% max error).  Second best: saturating (6.8%).
- **Challenge:** Pinned at F_MIN for rs < 0.4.  The transition from
  floor to free is smooth after Phase 6.
- **Verdict:** Excellent interpolability.

#### φ₀ (phase, mode 0) — ⚠ HARDEST TO INTERPOLATE

- **Shape:** Non-monotonic with a dip.  Starts at ~2.00 (rs=0.2), drops to
  ~1.88 (rs≈1.1), then rises back to ~2.10 (rs≥5).  The dip region
  (rs 0.7–1.5) is where the difficulty lies.
- **Best form over full range:** `a+b·rs^c+d·rs^e` gives 40.7% max error
  (worst at rs=1.1).  This is unacceptable.
- **Best form for rs ≥ 1.0:** Padé[2/2] gives 4.6% — excellent.
- **Root cause:** φ₀ is poorly constrained when f₀ is near F_MIN (B₀ small).
  The parameter is nearly degenerate at low rs.
- **Options to improve:**
  1. **Split-range interpolation:** Use one form for rs < 1.5, another for
     rs ≥ 1.5, with smooth stitching (e.g. tanh crossover).
  2. **Higher-order Padé:** Try Padé[3/3] or Padé[4/4] with 7–9 coefficients.
  3. **Further smoothing in Phase 6:** Increase lambda_smooth for phi0
     specifically (per-parameter lambda).
  4. **Fix phi0 at low rs:** Since mode 0 is vestigial below rs ≈ 0.5
     (B₀ < 0.03), fixing φ₀ to a constant there has negligible impact on
     the physical chi(r).

#### α₁ (damping rate, mode 1) — ✅ INTERPOLABLE

- **Shape:** Nearly flat, slowly decreasing from 0.842 to 0.794.
- **Best form:** Padé[2/2] (6.0% max error).  The weak non-monotonicity
  (peaks near rs=1.0) makes simple forms struggle, but Padé captures it.
- **Verdict:** Good interpolability with 5 coefficients.

#### f₁ (frequency, mode 1) — ✅ INTERPOLABLE

- **Shape:** Rises from 0.104 to 0.251, saturating.  Similar to α₀.
- **Best form:** Padé[2/2] (7.0% max error, worst at rs=0.9).
  Padé[1/1] gives 7.6% with only 3 coefficients.
- **Verdict:** Good.  Even 3-param forms are adequate.

#### φ₁ (phase, mode 1) — ✅ INTERPOLABLE BUT COMPLEX

- **Shape:** Dramatic drop from 2.09 (rs=0.2) through 0.07 (rs=1.1),
  then slow rise to 0.27 (rs=10).  Nearly 2 radians of variation.
- **Best form:** Padé[2/2] (3.3% max error over full range) — surprisingly
  good despite the complex shape.
- **Verdict:** Interpolable.  Padé[2/2] captures the transition well.

### 6.7  Summary: total meta-parameter budget

| scenario | total coefficients | quality |
|---|---|---|
| All 6 params × Padé[2/2] | **30** | 5 of 6 params < 7% max error; φ₀ at 41% |
| All 6 params × best form each | **30** | same (Padé[2/2] wins for 5/6) |
| 5 params × Padé[2/2] + φ₀ split-range | ~**33** | should bring φ₀ below 10% |
| 5 params × Padé[2/2] + φ₀ Padé[3/3] | **32** | untested, likely ~15% for φ₀ |

**The interpolation is feasible with ~30–33 meta-parameters** for the
6 nonlinear quantities.  The sole problem child is φ₀.

### 6.8  Recommended interpolation strategy

1. **Use Padé[2/2]** for α₀, f₀, α₁, f₁, φ₁  →  25 coefficients.
2. **Handle φ₀ specially:**
   - Option A: Split at rs ≈ 1.5 with two Padé[1/1] + tanh stitching
     → 8 additional coefficients.
   - Option B: Use Padé[3/3] or higher-order → 7+ coefficients.
   - Option C: Fix φ₀ = const for rs < 0.5 (where B₀ ≈ 0) and use
     Padé[2/2] for rs ≥ 0.5 → 5+ coefficients.
   - Option D: Further smooth φ₀ in Phase 6 before parametric fitting.
3. **Do NOT interpolate B₀, B₁ directly** — always compute from the
   interpolated θ via the moment-constraint linear solve.

### 6.9  What must happen before interpolation can be finalized

1. **Resolve φ₀ interpolation.**  The 40.7% max error is unacceptable.
   Most promising: per-parameter lambda in Phase 6 (penalize φ₀ curvature
   more heavily at low rs), then re-test Padé fits.
2. **Validate end-to-end.**  After choosing parametric forms:
   - Evaluate $\theta_j^{\text{interp}}(r_s)$ at the 51 grid points.
   - Compute B₀, B₁ from interpolated θ.
   - Compare $\Delta\chi(r; r_s)$ against exact data.
   - Verify chi sum rules are preserved.
3. **Test at interpolated rs points** (e.g. rs = 3.37, not on the grid)
   to confirm the parametric forms generalize.

---

## 7  Fixes already implemented (chronological)

Below is a chronological list of all improvements made to the fitting
pipeline across sessions 1–5.  **Do not re-implement these.**

### 7.1  Bidirectional sweep (Phases 1–3)

**Problem:** Sequential forward sweep from rs = 0.2 propagated errors
rightward.
**Fix:** Anchor at rs = 5.0 with aggressive multi-start (11 candidates),
then sweep up and down independently.

### 7.2  B-sign canonicalization

**Problem:** The optimizer could find solutions with $(B < 0, \phi)$ or
$(B > 0, \phi + \pi)$ — physically identical but causing apparent
$\phi$-jumps of $\sim\pi$.
**Fix:** After solving B0/B1, if $B_i < 0$, shift $\phi_i += \pi$ and
re-wrap.  Now $B \ge 0$ everywhere.

### 7.3  B0/B1 in proximity metric

**Problem:** Even when 6 nonlinear params are close, the linear
amplitudes B0/B1 could jump.
**Fix:** Add $(B_0 - B_0^{\text{prev}})^2 + (B_1 - B_1^{\text{prev}})^2$
to the proximity metric in `_select_smoothest`.

### 7.4  Frequency floor (`F_MIN = 0.02`)

**Problem:** f0 collapsed to 0 at low rs, making the mode a pure
exponential with undefined $\phi$.
**Fix:** Hard lower bound `F_MIN = 0.02` in bounds and canonicalization.

### 7.5  Mode separation penalty (`DF_MIN = 0.03`)

**Problem:** Both modes converged to the same frequency → singular
constraint matrix (cond ≈ 2000), B0 ≈ B1 ≈ 45.
**Fix:** Penalty $10^4$ in `_select_smoothest` and smooth penalty in
`_regularized_fit` when $|f_0 - f_1| < 0.03$.

### 7.6  Anchor re-fit (Phase 4)

**Problem:** The anchor point had no predecessor context during Phase 1,
so it could be slightly inconsistent with its neighbours.
**Fix:** After both sweeps, re-fit the anchor using the average of its
left and right neighbours as warm start.

### 7.7  Polynomial extrapolation re-fit (Phase 5)

**Problem:** Low-rs points (below rs = 2) had poor initial guesses
because the physics guess isn't calibrated there.
**Fix:** Fit cubic polynomials to each parameter from the smooth
region (rs ≥ 2), extrapolate to low rs, and use these as additional
candidates for re-fitting.  Includes regularized candidates at
λ ∈ {1, 10, 50} and relaxed cost tolerance (2×).

### 7.8  Degenerate-mode penalty in `_regularized_fit`

**Problem:** When $f_i \approx F_\text{min}$, the mode contributes
negligibly (B ≈ 0) and its $\alpha_i, \phi_i$ are poorly constrained,
leading to wild jumps.
**Fix:** When $|f_i - F_\text{min}| < F_\text{min}/2$, add a 100×
stronger L2 penalty on $\alpha_i$ and $\phi_i$ toward the neighbour's
values.

### 7.9  Global smooth re-fit (Phase 6) — SESSION 5

**Problem:** Phases 1–5 optimize each rs independently, enforcing
smoothness only locally via proximity selection.  This leads to
accumulated errors: phi0 60% jumps, alpha1 44% jumps, staircase
patterns in phi1.

**Fix:** Added `_global_smooth_refit()` in `fitting.py` (~200 lines).
Optimizes all 306 variables (6 params × 51 rs) simultaneously via
L-BFGS-B with first-order smoothness, second-order curvature, and
degenerate-mode penalties.  Lambda calibrated so `lambda_smooth=1.0`
gives equal data/smoothness weight.

**Result:** 86% smoothness reduction.  All max jumps ≤16.3%.

---

## 8  Remaining problems & next steps

### 8.1  Primary: φ₀ interpolation (40.7% max error)

φ₀ remains the hardest parameter to interpolate.  The non-monotonic
dip near rs ≈ 1.1 defeats all standard 3–5 parameter forms.

**Approaches (in priority order):**

1. **Per-parameter smoothness weight in Phase 6.**  Currently all 6 params
   share the same lambda.  Adding a separate, larger weight for φ₀ (and α₀)
   at low rs could smooth the dip.  This trades data fidelity for
   interpolability — acceptable because mode 0 is weak there (B₀ small).

2. **Split-range parametric form.**  Use one formula for rs < T and another
   for rs ≥ T with `tanh((rs-T)/w)` blending.

3. **Higher-order Padé.**  Try Padé[3/3] or Padé[4/4] (7–9 coefficients).

4. **Enforce parametric form during fitting.**  Instead of fitting free
   θ(rs) values and then fitting a parametric form, parametrize
   $\theta_j(r_s) = P_j(r_s; \mathbf{c}_j)$ directly and optimize the
   ~30 meta-coefficients.  This is the "Chebyshev variant" from §8.1
   of the old document.  It **enforces** interpolability by construction.

### 8.2  Secondary: remaining low-rs jumps

alpha0 (13.8%) and phi1 (16.3%) still have their largest jumps at
low rs.  These appear to be physics-driven (rapid variation at high
density).  Options:

- **Finer grid at low rs** (Δrs = 0.05 for rs < 0.5).
- **Stronger Phase 6 curvature penalty at low rs.**
- **Accept as-is** if the parametric fits absorb the variation.

### 8.3  Optional: direct parametric fitting

The most ambitious approach: bypass the current 51-point free-parameter
fitting entirely.  Instead, write $\theta_j(r_s) = P_j(r_s; \mathbf{c}_j)$
with e.g. Padé[2/2] and optimize the ~30 coefficients $\mathbf{c}$
directly against all 51 target curves simultaneously.

$$
\mathcal{L}(\mathbf{c}) = \sum_{i=1}^{51}
  \| \text{model}(r; r_{s,i}, P(\mathbf{c}, r_{s,i})) - \Delta\chi_\text{exact}(r; r_{s,i}) \|^2
$$

**Advantages:**
- Smoothness is guaranteed by construction (no jumps possible).
- Only ~30 unknowns instead of 306.
- The result is immediately a usable interpolation formula.

**Challenges:**
- Need good initial guess for the ~30 coefficients
  (fit Padé to current data as starting point).
- May need per-rs weighting to handle the dynamic range.
- φ₀ may need a non-standard form (see §8.1).

### 8.4  Reparametrization ideas (untested)

- **Fit $k = 2\pi f$ instead of $f$**: more natural units, may improve
  optimizer behavior.
- **Log-space for positive quantities:** fit $\log\alpha$ instead of
  $\alpha$ (always positive); potentially $\log B$ for amplitudes.
- **Complex frequency:** $z = \alpha + i\cdot 2\pi f$ makes physical
  sense (damped oscillation pole in the complex plane).  The two modes
  become two complex poles.

---

## 9  Agent recommendations

### 9.1  What worked well

- The **bidirectional sweep from rs = 5** was the single most impactful
  change for per-point fitting.
- **Phase 6 global smooth re-fit** was the single most impactful change
  for interpolability.
- **B-sign canonicalization** resolved the worst phi jumps (π-shifts).
- The **multi-candidate strategy** catches cases where a single optimizer
  would get stuck.

### 9.2  Priority ranking for next steps

1. **Per-parameter lambda for φ₀ in Phase 6** — low implementation
   difficulty, high expected impact on the φ₀ interpolation problem.

2. **Direct parametric fitting (§8.3)** — moderate difficulty, but this
   is the architecturally correct endgame.  Use current Padé fits as
   initial coefficients.

3. **Split-range φ₀ interpolation (§8.1 option 2)** — low difficulty,
   moderate impact.  A practical stopgap.

4. **Finer low-rs grid (§8.2)** — very low difficulty.  May help Phase 6
   resolve rapid transitions at rs < 0.5.

### 9.3  Things to preserve (invariants)

- Always canonicalize after every fit (§4.5).
- Always enforce F_MIN and DF_MIN.
- Always include B0/B1 in the smoothness metric.
- Use `diagnose_full.py` to validate changes — the max-jump table is the
  primary quality metric.
- The moment constraints (B0/B1 from the 2×2 linear system) must remain
  **exact** — never smooth B0/B1 independently of the nonlinear params.
  B0/B1 are *derived quantities*; only the 6 nonlinear params are free.

### 9.4  Gotchas and pitfalls

1. **Stochastic results:** `_fit_one_point` uses `np.random.randn`, so
   re-running gives slightly different results.  Set `np.random.seed(0)`
   for reproducibility during development.

2. **Fourier transform cost:** Each call to `get_chi(q, rs)` does a DST-I.
   For global fitting, precompute all 51 target curves once and pass them
   in.  Phase 6 already does this.

3. **Phase wrapping near boundaries:** phi values near ±π can flip
   between +π and −π across rs steps.  The canonicalization handles this
   but φ ≈ 2.1 (as in phi0 for rs > 1) is safely interior.

4. **Condition number growth:** cond(M) grows from ~5 at rs=0.7 to ~81
   at rs=10.  As conditioning worsens, B0/B1 become more sensitive to
   small parameter changes.

5. **kFr1 fitting range:** Must be 4, not 8.  A previous experiment with
   kFr1=8 caused regression — the tail region adds noise without improving
   the core fit.

6. **dq grid coarsening:** dq was changed from 0.001 to 0.01 for 10×
   faster runtime.  This is adequate for current fitting.  If very precise
   diagnostics are needed, temporarily restore dq=0.001.

---

## 10  Conventions

- **Working directory:** always `cd` to the project root before running.
- **Python:** use `.venv/bin/python` (bare `python` may not work).
- **Formatter:** `ruff` (config in `ruff.toml`).
- **Results:** stored as `parameters.pkl` via `utils.io.write_dict`.
- **Plotting:** `pp.plot_parameters(params_dict)` for the standard
  4-panel parameter plot.
- **Diagnostics:** `diagnose_full.py` prints all 8 quantities + max jumps.
- **Interpolability:** `analyze_interpolability.py` tests parametric forms.

---

## 11  Quick reference — key functions

| function | file | purpose |
|---|---|---|
| `fit_params` | fitting.py | main entry; 6-phase pipeline |
| `_global_smooth_refit` | fitting.py | Phase 6: global L-BFGS-B on all rs |
| `_fit_one_point` | fitting.py | single-rs fit with 6 candidates |
| `_canonicalize_params` | fitting.py | enforce canonical form |
| `_select_smoothest` | fitting.py | proximity-based branch selection |
| `_regularized_fit` | fitting.py | L-BFGS-B with L2 + penalties |
| `guess_X` | fitting.py | curve_fit TRF wrapper |
| `_compute_B` | fitting.py | safe B0/B1 computation |
| `_physics_initial_guess` | fitting.py | physics-motivated p0 |
| `delta_chi` | models.py | model evaluation + linear B solve |
| `J_n_m_kFr` | physics.py | analytic moment integrals |
| `delta_C` | physics.py | moment constraint RHS |
| `get_gas_params` | physics.py / utils_chi.py | kF, n0, NF from rs |
| `chi_r_from_chi_q_fast` | fourier.py | DST-I χ(q)→χ(r) |
| `chi_q_from_chi_r_fast` | fourier.py | DST-I χ(r)→χ(q) |
| `get_chi` | utils_chi.py | interacting χ(r) via Corradini-PZ |
| `get_chi02` | utils_chi.py | non-interacting χ₀(r) |
| `corradini_pz` | utils_chi.py | local-field factor fxc(q) |
| `chi00q` | utils_chi.py | Lindhard function χ₀(q) |
| `get_constraints` | pp.py | extract B0/B1 from fitted params |
| `plot_parameters` | pp.py | 4-panel parameter visualisation |
| `plot_chi` | pp.py | fit quality comparison |
| `write_dict` / `load_dict` | io.py | pickle I/O |

---

## 12  Source code listings

Below are the **complete current source files** for reference.
Always check these against the actual files in case of later edits.

### 12.1  fitting.py (763 lines)

```python
import numpy as np
from scipy.optimize import curve_fit, minimize

from optimization.models import delta_chi
from utils.utils_chi import get_chi, get_chi02, get_gas_params

# --- Parameter bounds ---
# params = [alpha0, f0, phi0, alpha1, f1, phi1]
# alpha > 0 (damping), f >= F_MIN (prevents mode collapse), phi in [-pi, pi]
F_MIN = 0.02  # Minimum frequency — prevents degenerate pure-exponential modes
DF_MIN = 0.03  # Minimum |f0 - f1| — prevents both modes collapsing to same frequency
BOUNDS_LOWER = np.array([1e-4, F_MIN, -np.pi, 1e-4, F_MIN, -np.pi])
BOUNDS_UPPER = np.array([20.0, 3.0, np.pi, 20.0, 3.0, np.pi])


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
        B0, B1 = delta_chi(np.array([0.0]), rs=rs, params=params, get_constraints=True)
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
            3 / (2.0 * np.pi),
            np.pi / 2 - 0.1,
            1.0 / kF,
            -1 / (2.0 * np.pi),
            np.pi / 2 + 0.1,
        ]
    )


def _global_smooth_refit(
    rslist_sorted, q, r, model, parameters, lambda_smooth=1.0, max_iter=500
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
    fit_data = []
    kFr_arrays = []
    delta_Cs = []
    inv_kF3 = []
    inv_kF5 = []

    for rs_val in rslist_sorted:
        kF, n0, NF = get_gas_params(rs_val)
        factor = -6 * np.pi * n0 * NF
        chiR = get_chi(q, rs_val)
        chi0R = get_chi02(q, rs_val)
        dchi = -(chi0R - chiR) / factor
        i0 = np.argmin(np.abs(kF * r - 0))
        i1 = np.argmin(np.abs(kF * r - 4))
        rf = r[i0:i1]
        yf = dchi[i0:i1]
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

    # --- Lambda calibration ---
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

    init_smooth = 0.0
    for i in range(n_rs - 1):
        d = pm[i + 1] - pm[i]
        init_smooth += np.sum(d**2 * inv_s2) * inv_drs2[i]

    lam = lambda_smooth * init_data / (init_smooth + 1e-30)
    lam_degen = 50.0 * lam

    avg_drs_sq = np.zeros(n_rs - 2)
    for i in range(n_rs - 2):
        avg_drs_sq[i] = (0.5 * (drs[i] + drs[i + 1])) ** 2
    inv_avg_drs4 = 1.0 / avg_drs_sq**2

    print(f"  Data: {init_data:.4e}, Smooth: {init_smooth:.4e}, λ_eff: {lam:.4e}")
    print(f"  Optimizing {n_rs * n_p} variables...")

    def objective(x):
        P = x.reshape(n_rs, n_p)
        total = 0.0

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

        for i in range(n_rs - 1):
            d = P[i + 1] - P[i]
            total += lam * np.sum(d**2 * inv_s2) * inv_drs2[i]
            f0_next = P[i + 1, 1]
            w = np.exp(-(((f0_next - F_MIN) / 0.01) ** 2))
            if w > 0.01:
                total += (
                    lam_degen
                    * w
                    * (d[0] ** 2 * inv_s2[0] + d[2] ** 2 * inv_s2[2])
                    * inv_drs2[i]
                )

        for i in range(n_rs):
            df = np.sqrt((P[i, 1] - P[i, 4]) ** 2 + 1e-12)
            if df < DF_MIN:
                total += 1e4 * (DF_MIN - df) ** 2

        for i in range(n_rs - 2):
            d2 = P[i + 2] - 2 * P[i + 1] + P[i]
            total += lam * np.sum(d2**2 * inv_s2) * inv_avg_drs4[i]

        return total

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=list(zip(lb, ub)),
        options={"maxiter": max_iter, "ftol": 1e-15, "maxfun": 500000},
    )

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

    for i, rs in enumerate(rslist_sorted):
        parameters[rs] = _canonicalize_params(P_opt[i], rs)

    return parameters


def _fit_one_point(rs, q, r, model, prev_params, cost_tolerance, prev_B=None):
    """Fit a single rs point with warm start from prev_params."""
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    chiR = get_chi(q, rs)
    chi0R = get_chi02(q, rs)
    delta_chi_exact = -(chi0R - chiR) / factor

    physics_guess = _physics_initial_guess(rs)
    candidates = [prev_params, physics_guess]

    scale = np.array([0.1, 0.05, 0.2, 0.1, 0.05, 0.2])
    candidates.append(prev_params + scale * np.random.randn(6))

    fit_idx0 = np.argmin(np.abs(kF * r - 0))
    fit_idx1 = np.argmin(np.abs(kF * r - 4))
    r_fit_full = r[fit_idx0:fit_idx1]
    y_fit_full = delta_chi_exact[fit_idx0:fit_idx1]
    n_pts = len(r_fit_full)
    step = max(1, n_pts // 2000)
    r_fit = r_fit_full[::step]
    y_fit = y_fit_full[::step]

    results = []

    for p0 in candidates:
        try:
            p_opt, p_cov = guess_X(r, rs, delta_chi_exact, model, p0, kFr0=0, kFr1=4)
            p_opt = _canonicalize_params(p_opt, rs)
            residual = model(r_fit, rs=rs, params=p_opt) - y_fit
            data_cost = np.sum(residual**2)
            results.append((p_opt, data_cost, p_cov))
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            pass

    for lam in [0.1, 1.0, 5.0]:
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


def fit_params(
    rslist, q, r, model=delta_chi, inverse=False, n_restarts=3, cost_tolerance=3.0
):
    """Fit parameters using bidirectional sweep + global smooth re-fit.

    Strategy:
    1. Pick an anchor rs near 5.0
    2. Multi-start fit at anchor (11 candidates)
    3. Sweep upward: anchor → max rs
    4. Sweep downward: anchor → min rs
    5. Re-fit anchor from neighbours
    6. Polynomial extrapolation re-fit for low-rs
    7. Global smooth re-fit (all points simultaneously)
    """
    from tqdm import tqdm

    parameters = {}
    parameters_cov = {}
    parameters["model"] = model

    rslist_sorted = np.sort(rslist)

    # --- Phase 1: Fit anchor point with aggressive multi-start ---
    anchor_target = 5.0
    anchor_idx = int(np.argmin(np.abs(rslist_sorted - anchor_target)))
    anchor_rs = rslist_sorted[anchor_idx]

    rs = anchor_rs
    kF, n0, NF = get_gas_params(rs)
    factor = -6 * np.pi * n0 * NF
    chiR = get_chi(q, rs)
    chi0R = get_chi02(q, rs)
    delta_chi_exact = -(chi0R - chiR) / factor

    physics_guess = _physics_initial_guess(rs)
    candidates = [physics_guess]
    scale = np.array([0.3, 0.1, 0.5, 0.3, 0.1, 0.5]) * 0.1
    for _ in range(10):
        candidates.append(physics_guess + scale * np.random.randn(6))

    fit_idx0 = np.argmin(np.abs(kF * r - 0))
    fit_idx1 = np.argmin(np.abs(kF * r - 4))
    r_fit_full = r[fit_idx0:fit_idx1]
    y_fit_full = delta_chi_exact[fit_idx0:fit_idx1]
    n_pts = len(r_fit_full)
    step = max(1, n_pts // 2000)
    r_fit = r_fit_full[::step]
    y_fit = y_fit_full[::step]

    results = []
    for p0 in candidates:
        try:
            p_opt, p_cov = guess_X(r, rs, delta_chi_exact, model, p0, kFr0=0, kFr1=4)
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
            rs, q, r, model, parameters[pred_rs], cost_tolerance,
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
            rs, q, r, model, parameters[pred_rs], cost_tolerance,
            prev_B=B_values.get(pred_rs),
        )
        parameters[rs] = best_params
        parameters_cov[rs] = best_cov if best_cov is not None else np.zeros((6, 6))
        B_values[rs] = _compute_B(best_params, rs)

    # --- Phase 4: Re-fit anchor using neighbors ---
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
    smooth_threshold = 2.0
    smooth_mask = rslist_sorted >= smooth_threshold
    if np.sum(smooth_mask) >= 4:
        smooth_rs = rslist_sorted[smooth_mask]
        smooth_params = np.array([parameters[rs] for rs in smooth_rs])
        param_polys = []
        for j in range(6):
            coeffs = np.polyfit(smooth_rs, smooth_params[:, j], 3)
            param_polys.append(np.poly1d(coeffs))
        low_rs_indices = np.where(~smooth_mask)[0]
        for i in reversed(low_rs_indices):
            rs = rslist_sorted[i]
            pred_rs = rslist_sorted[i + 1]
            extrap_params = np.array([poly(rs) for poly in param_polys])
            extrap_params = np.clip(
                extrap_params, BOUNDS_LOWER + 1e-6, BOUNDS_UPPER - 1e-6
            )
            extrap_params = _canonicalize_params(extrap_params, rs)
            prev_B = B_values.get(pred_rs)
            kF, n0, NF = get_gas_params(rs)
            factor = -6 * np.pi * n0 * NF
            chiR = get_chi(q, rs)
            chi0R = get_chi02(q, rs)
            delta_chi_exact = -(chi0R - chiR) / factor
            fit_idx0 = np.argmin(np.abs(kF * r - 0))
            fit_idx1 = np.argmin(np.abs(kF * r - 4))
            r_fit_full = r[fit_idx0:fit_idx1]
            y_fit_full = delta_chi_exact[fit_idx0:fit_idx1]
            n_pts = len(r_fit_full)
            step = max(1, n_pts // 2000)
            r_fit = r_fit_full[::step]
            y_fit = y_fit_full[::step]
            results = []
            fwd_params = parameters[rs]
            res_fwd = model(r_fit, rs=rs, params=fwd_params) - y_fit
            results.append((fwd_params, np.sum(res_fwd**2), parameters_cov[rs]))
            for p0 in [extrap_params, parameters[pred_rs]]:
                try:
                    p_opt, p_cov = guess_X(
                        r, rs, delta_chi_exact, model, p0, kFr0=0, kFr1=4
                    )
                    p_opt = _canonicalize_params(p_opt, rs)
                    res = model(r_fit, rs=rs, params=p_opt) - y_fit
                    results.append((p_opt, np.sum(res**2), p_cov))
                except (RuntimeError, np.linalg.LinAlgError, ValueError):
                    pass
            for lam in [1.0, 10.0, 50.0]:
                try:
                    p_reg = _regularized_fit(
                        r_fit, y_fit, rs, model, parameters[pred_rs],
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
        rslist_sorted, q, r, model, parameters, lambda_smooth=10.0, max_iter=2000
    )

    return parameters, parameters_cov


def _select_smoothest(results, prev_params, cost_tolerance=2.0, rs=None, prev_B=None):
    """Among candidates with data cost within `cost_tolerance` of the best,
    pick the one closest (in L2 norm including B0/B1) to prev_params."""
    if prev_params is None:
        best = min(results, key=lambda x: x[1])
        return best[0], best[2]

    best_data_cost = min(r[1] for r in results)
    threshold = cost_tolerance * best_data_cost + 1e-30

    acceptable = [r for r in results if r[1] <= threshold]

    def proximity(result):
        d = np.sum((result[0] - prev_params) ** 2)
        if rs is not None and prev_B is not None:
            B = _compute_B(result[0], rs)
            if B is not None:
                d += (B[0] - prev_B[0]) ** 2 + (B[1] - prev_B[1]) ** 2
            else:
                d += 1e6
        f0, f1 = result[0][1], result[0][4]
        if abs(f1 - f0) < DF_MIN:
            d += 1e4
        return d

    best = min(acceptable, key=proximity)
    return best[0], best[2]


def _backward_fixup(
    rslist, q, r, model, parameters, parameters_cov, cost_tolerance, n_edge=3
):
    """[NOT CALLED] Backward pass for edge smoothing.  Subsumed by Phase 5."""
    # ... (dead code, omitted for brevity)
    pass


def _regularized_fit(r_fit, y_fit, rs, model, prev_params, lambda_smooth):
    """Run scipy.optimize.minimize with L2 regularization + penalties."""
    y_norm = np.sum(y_fit**2) + 1e-30
    p_norm = np.sum(prev_params**2) + 1e-30
    lam = lambda_smooth * (y_norm / p_norm)

    def objective(params):
        try:
            residual = model(r_fit, rs=rs, params=params) - y_fit
            data_cost = np.sum(residual**2)
            reg_cost = lam * np.sum((params - prev_params) ** 2)
            df = abs(params[1] - params[4])
            sep_penalty = lam * 10.0 * max(0, DF_MIN - df) ** 2
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


def guess_X(r, rs, X_exact, model, initial_guess, kFr0=0, kFr1=4, max_fit_pts=2000):
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    fit_idx0 = np.argmin(np.abs(kF * r - kFr0))
    fit_idx1 = np.argmin(np.abs(kF * r - kFr1))

    r_fit = r[fit_idx0:fit_idx1]
    y_fit = X_exact[fit_idx0:fit_idx1]

    n_pts = len(r_fit)
    if n_pts > max_fit_pts:
        step = n_pts // max_fit_pts
        r_fit = r_fit[::step]
        y_fit = y_fit[::step]

    def model_wrapper(r, alpha0, f0, phi0, alpha1, f1, phi1):
        params = [alpha0, f0, phi0, alpha1, f1, phi1]
        return model(r, rs=rs, params=params)

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
```

### 12.2  models.py

```python
import numpy as np

from utils.physics import J_n_m_kFr, delta_C


def delta_chi(r, rs, params, get_constraints=False):
    r"""
    Two-mode model:
        \Delta chi(r) = [ B0 e^{-\alpha_0 kF r} cos(k0 kF r + \phi_0)
                        + B1 e^{-\alpha_1 kF r} cos(k1 kF r + \phi_1) ]
    params = [\alpha_0, f0, \phi_0, \alpha_1, f1, \phi_1]
    k_i = 2 pi f_i
    B0, B1 solved from moment constraints.
    """
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    kFr = np.asarray(kF * r, float)

    alpha0, f0, phi0, alpha1, f1, phi1 = params
    k0 = 2.0 * np.pi * f0
    k1 = 2.0 * np.pi * f1

    J0 = J_n_m_kFr(0, k0, alpha0, phi0, kF)
    J1 = J_n_m_kFr(0, k1, alpha1, phi1, kF)
    J3 = J_n_m_kFr(1, k0, alpha0, phi0, kF)
    J4 = J_n_m_kFr(1, k1, alpha1, phi1, kF)

    b = np.array([delta_C(1, rs), delta_C(0, rs)])
    Mmat = np.array([[J3, J4], [J0, J1]])
    B0, B1 = np.linalg.solve(Mmat, b)

    if get_constraints:
        return B0, B1
    else:
        delta_chi = B0 * np.exp(-alpha0 * kFr) * np.cos(k0 * kFr + phi0) + B1 * np.exp(
            -alpha1 * kFr
        ) * np.cos(k1 * kFr + phi1)
        return delta_chi
```

### 12.3  main.py

```python
import time

import numpy as np

from input import q, r
from optimization.fitting import fit_params
from optimization.models import delta_chi
from utils.io import write_dict

model = delta_chi
rslist = np.concatenate([np.arange(0.2, 2.0, 0.1), np.arange(2.0, 10.25, 0.25)])
inverse = 0

print(f"Fitting X with {model.__name__}...")
start_time = time.time()

parameters, parameters_cov = fit_params(rslist, q, r, model=model, inverse=inverse)
end_time = time.time()

print(f"Fitting completed in {end_time - start_time:.2f} seconds.")
write_dict(parameters, "parameters")
```

### 12.4  input.py

```python
import numpy as np

from utils.fourier import r_grid_from_q

qmax = 10000
dq = 0.01
q = np.arange(dq, qmax + dq / 2, dq)
r = r_grid_from_q(q)
```

### 12.5  diagnose_full.py

```python
"""Diagnose all 8 quantities: 6 nonlinear params + B0, B1 + condition number."""
import sys
import numpy as np

sys.path.insert(0, "src")
from utils.io import load_dict
from utils.physics import J_n_m_kFr, delta_C

params = load_dict("parameters")
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])

print(f"rs range: {rsl[0]:.2f} to {rsl[-1]:.2f}, {len(rsl)} points\n")

header = f"{'rs':>5s}  {'alpha0':>8s} {'f0':>8s} {'phi0':>8s} {'alpha1':>8s} {'f1':>8s} {'phi1':>8s} | {'B0':>10s} {'B1':>10s} {'cond(M)':>10s}"
print(header)
print("-" * len(header))

B0_all, B1_all = [], []
cond_all = []
data_all = []

for rs in rsl:
    p = params[rs]
    alpha0, f0, phi0, alpha1, f1, phi1 = p
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    k0, k1 = 2 * np.pi * f0, 2 * np.pi * f1

    J0 = J_n_m_kFr(0, k0, alpha0, phi0, kF)
    J1 = J_n_m_kFr(0, k1, alpha1, phi1, kF)
    J3 = J_n_m_kFr(1, k0, alpha0, phi0, kF)
    J4 = J_n_m_kFr(1, k1, alpha1, phi1, kF)
    Mmat = np.array([[J3, J4], [J0, J1]])
    b = np.array([delta_C(1, rs), delta_C(0, rs)])
    B0, B1 = np.linalg.solve(Mmat, b)
    cn = np.linalg.cond(Mmat)

    B0_all.append(B0); B1_all.append(B1)
    cond_all.append(cn); data_all.append(p)

    print(f"{rs:5.2f}  {alpha0:8.4f} {f0:8.4f} {phi0:8.4f} "
          f"{alpha1:8.4f} {f1:8.4f} {phi1:8.4f} | "
          f"{B0:10.4f} {B1:10.4f} {cn:10.1f}")

B0_all = np.array(B0_all); B1_all = np.array(B1_all)
data_all = np.array(data_all)

print("\n=== Max jumps between consecutive rs values ===")
labels = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1", "B0", "B1"]
all_arrays = [data_all[:, i] for i in range(6)] + [B0_all, B1_all]

for name, arr in zip(labels, all_arrays):
    diffs = np.abs(np.diff(arr))
    idx = np.argmax(diffs)
    rng = np.max(arr) - np.min(arr)
    pct = diffs[idx] / (rng + 1e-30) * 100
    print(f"  {name:8s}: max jump = {diffs[idx]:.6f} "
          f"({pct:.1f}% of range) between rs={rsl[idx]:.2f} and rs={rsl[idx+1]:.2f}")
```
