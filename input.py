import numpy as np

from utils.fourier import r_grid_from_q

qmax = 10000
dq = 0.001
q = np.arange(dq, qmax + dq / 2, dq)  # starts at 0; code will drop q=0 internally
r = r_grid_from_q(q)
