import copy
import numpy as np
from control import run
from params import params

def dt_conv_test(dt_vals):
    error = np.zeros_like(dt_vals)
    params['plot'] = False
    for i in range(dt_vals.shape[0]):
        p = copy.deepcopy(params)
        dt = dt_vals[i]
        p['delta_t'] = dt
        error[i] = run()
    print error


def dx_conv_test():
    pass

t_max = params['t_max']
n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
dt_list = np.array([t_max / n for n in n_list])
dt_conv_test(dt_list)
