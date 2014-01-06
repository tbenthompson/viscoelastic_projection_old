from control import run
from params import params

params['plot'] = False
params['t_max'] = 100.0 * 3600 * 24 * 365.0
params['adapt_tol'] = 5e-4
params['delta_t'] = params['t_max'] / 16.0
params['t_max'] = params['t_max'] / 1.0
run()
