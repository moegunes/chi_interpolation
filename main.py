import time

import numpy as np

from input import q, r
from optimization.fitting import fit_params
from optimization.models import delta_chi_nl
from utils.io import write_dict

model = delta_chi_nl
# gamma = 1

rslist = np.arange(0.25, 10.25, 0.25)
inverse = 0

print(f"Fitting X with {model.__name__}...")
start_time = time.time()

parameters, parameters_cov = fit_params(rslist, q, r, model=model, inverse=inverse)
end_time = time.time()

print(f"Fitting completed in {end_time - start_time:.2f} seconds.")
write_dict(parameters, "parameters")
