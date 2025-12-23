import numpy as np


def get_constraints(params_dict, rslist):
    model = params_dict["model"]
    gamma = params_dict["gamma"]
    C0_list = []
    C1_list = []
    D0_list = []
    D1_list = []

    for rs in rslist:
        p_opt = params_dict[rs]
        C0, D0, C1, D1 = model(
            r=0, rs=rs, params=p_opt, gamma=gamma, get_constraints=True
        )  # r=0 is a dummy value
        C0_list.append(C0)
        C1_list.append(C1)
        D0_list.append(D0)
        D1_list.append(D1)
    return np.array(C0_list), np.array(D0_list), np.array(C1_list), np.array(D1_list)
