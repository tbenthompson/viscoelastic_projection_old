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
        params['t_max'] = dt
        error[i] = run()
    print error


def dx_conv_test():
    pass

t_max = params['t_max']
n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]#, 32, 64, 128, 256]
dt_list = np.array([t_max / n for n in n_list])
dt_conv_test(dt_list)
# [ 0.20627478  0.06342581  0.02465175  0.0150974   0.01749603  0.02251856
#   0.02324016  0.02428559  0.02529668]
# [ 0.20805046  0.06368885  0.02538848  0.01262112  0.01581128  0.01940952
#   0.0219564   0.02304429  0.02342397]

# These were calculated using one time step which got shorter,
# n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# t_max = 100 years / n
# dt = 100 years / n
# [ 0.20627478  0.09604368  0.05368977  0.02877043  0.01455314  0.00747645
  # 0.00384754  0.00210327  0.00106475]
