import copy
import numpy as np
from control import run
from params import params

def dt_conv_test(dt_vals):
    error = np.zeros_like(dt_vals)
    params['plot'] = False
    for i in range(dt_vals.shape[0]):
        dt = dt_vals[i]
        params['delta_t'] = dt
        # params['t_max'] = dt
        error[i] = run()
    print error


def dx_conv_test():
    pass

params['adapt_tol'] = 1e-4
t_max = params['t_max']
n_list = [16, 32]#, 64, 128, 256]
dt_list = np.array([t_max / n for n in n_list])
dt_conv_test(dt_list)
# These were calculated using one time step which got shorter,
# t_max = 100 years / n
n_list = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
dt = t_max / n_list
error = np.array([0.20627478, 0.09604368, 0.05368977, 0.02877043, 0.01455314, 0.00747645, 0.00384754, 0.00210327, 0.00106475])
# These calculated with full time!
# n_list = [1, 2, 4, 8, 16, 32]
# t_max = 100 years
# dt = 100 years / n
# [ 0.20596201  0.09331718  0.06018622  0.03239035  0.03233129  0.03617945]
