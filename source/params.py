from material import wetdiabase

secs_in_a_year = 3600 * 24 * 365
params = dict()

#what material to use
params['material'] = wetdiabase

#time stepping
params['delta_t'] = 1.0 * secs_in_a_year
params['t_max'] = 100.0 * secs_in_a_year

#grid descriptors
params['x_min'] = 10.0
params['x_max'] = 2.0e4
params['y_min'] = 0.0
params['y_max'] = 2.0e4
params['x_points'] = 20
params['y_points'] = 20

# Far field plate rate boundary condition.
params['plate_rate'] = (40.0 / 1.0e3) / secs_in_a_year  # 40 mm/yr

# Initial stress setup -- fed into an elastic half-space solution
# to determine initial conditions. In the future, I could numerically
# solve a Poisson problem to determine a solution that would allow
# slip variations and elastic modulus variations.
params['fault_slip'] = 2.0
params['fault_depth'] = 1.0e4
params['elastic_depth'] = 1.0e4
params['viscosity'] = 5.0e19

# Where to save data?
params['run_name'] = 'test'
