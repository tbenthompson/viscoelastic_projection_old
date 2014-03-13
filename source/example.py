from control import run
from params import params

params['plot'] = True
# params['adaptive_mesh'] = False
# params['all_steps_adaptive'] = False
params['x_min'] = 10.0
# params['x_points'] = 200
# params['y_points'] = 200
params['t_max'] = 100.0 * 3600 * 24 * 365.0
params['adapt_tol'] = 1e-5
params['delta_t'] = params['t_max'] / 128.0
params['t_max'] = params['t_max'] / 1.0
run()
