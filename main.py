import time

import numpy as np

from input import q, r
from optimization.fitting import fit_params
from optimization.models import X_r2_two_mode
from utils.io import write_dict

model = X_r2_two_mode
gamma = 2

rslist = np.arange(2, 10.25, 0.25)
inverse = False
fit_residue = "hybrid"

print(f"Fitting X with {model.__name__}...")
start_time = time.time()

parameters, parameters_cov = fit_params(
    rslist, q, r, model=model, inverse=inverse, gamma=gamma, fit_residue=fit_residue
)
end_time = time.time()

print(f"Fitting completed in {end_time - start_time:.2f} seconds.")
write_dict(parameters, "parameters")
