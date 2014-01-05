from control import run
from params import params

params['plot'] = False
params['t_max'] = 100.0 * 3600 * 24 * 365.0
params['delta_t'] = params['t_max'] / 100.0
run()
