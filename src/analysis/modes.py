import numpy as np


def get_constraints(params_dict, rslist):
    model = params_dict["model"]
    gamma = params_dict["gamma"]
    B0_list = []
    B1_list = []

    for rs in rslist:
        p_opt = params_dict[rs]
        B0, B1 = model(
            r=0, rs=rs, params=p_opt, gamma=gamma, get_constraints=True
        )  # r=0 is a dummy value
        B0_list.append(B0)
        B1_list.append(B1)
    return np.array(B0_list), np.array(B1_list)
