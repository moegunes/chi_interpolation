"""Compare chi fit quality between parameter sets."""

import sys

import numpy as np

sys.path.insert(0, "src")
from input import q, r
from optimization.production import get_chi_interp
from utils.fourier import chi_q_from_chi_r_fast
from utils.io import load_dict
from utils.utils_chi import get_chi, get_gas_params

param_files = ["parameters", "parameters_lambda50", "parameters_pade_constrained"]

print("Chi(q) fit quality comparison:")
print("=" * 70)

for pfile in param_files:
    try:
        params = load_dict(pfile)
        rsl = sorted(
            [k for k in params.keys() if isinstance(k, (float, int)) and k != "model"]
        )
        print(f"\n{pfile}: ({len(rsl)} rs points)")

        errors = []
        for rs in [1.0, 2.0, 4.0, 8.0]:
            if rs not in rsl:
                print(f"  rs={rs:.1f}: not in dataset")
                continue
            kF, n0, NF = get_gas_params(rs)
            chi_target = get_chi(q, rs)
            chi_r = get_chi_interp(r, q, params, rs)
            _, chi_q = chi_q_from_chi_r_fast(r, chi_r, qlist=q)

            mask = (q > 0.5 * kF) & (q < 4 * kF)
            rel_err = (
                np.max(np.abs(chi_q[mask] - chi_target[mask]))
                / np.max(np.abs(chi_target[mask]))
                * 100
            )
            errors.append(rel_err)
            print(f"  rs={rs:.1f}: max rel err = {rel_err:.2f}%")

        if errors:
            print(f"  Average: {np.mean(errors):.2f}%")
    except Exception as e:
        import traceback

        print(f"  {pfile}: Error - {e}")
        traceback.print_exc()
