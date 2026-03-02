import sys

import numpy as np

sys.path.insert(0, "src")
from utils.io import load_dict

params = load_dict("parameters")
rsl = sorted([k for k in params.keys() if isinstance(k, (float, int)) and k != "model"])
data = np.array([params[rs] for rs in rsl])

names = ["alpha0", "f0", "phi0", "alpha1", "f1", "phi1"]
header = f"{'rs':>6s}  " + "  ".join(f"{n:>10s}" for n in names)
print(header)
print("-" * 80)
for i, rs in enumerate(rsl):
    p = data[i]
    line = f"{rs:6.2f}  " + "  ".join(f"{v:10.4f}" for v in p)
    print(line)

print()
print("=== Max jumps between consecutive rs values ===")
diffs = np.abs(np.diff(data, axis=0))
ranges = np.ptp(data, axis=0)
for j, name in enumerate(names):
    idx = np.argmax(diffs[:, j])
    jump = diffs[idx, j]
    pct = 100 * jump / ranges[j] if ranges[j] > 0 else 0
    print(
        f"  {name:>8s}: max jump = {jump:.4f} ({pct:5.1f}% of range) between rs={rsl[idx]:.2f} and rs={rsl[idx + 1]:.2f}"
    )
