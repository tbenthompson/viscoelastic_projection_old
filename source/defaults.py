import copy
from material import wetdiabase
from analytic_fast import simple_stress

secs_in_a_year = 3600 * 24 * 365.0
defaults = dict()

# what material to use
defaults['material'] = wetdiabase

# time stepping
# defaults['delta_t'] = 0. * secs_in_a_year
defaults['t_max'] = 100.0 * secs_in_a_year
defaults['delta_t'] = defaults['t_max'] / 1.0

# Meshing descriptors.
defaults['x_min'] = 100.0
defaults['x_max'] = 10.0e4
defaults['y_min'] = 0.0
defaults['y_max'] = 5.0e4
defaults['x_points'] = 10
defaults['y_points'] = 10
#saved_mesh.h5 has x_min=1.0
#saved_mesh2.h5 has x_min=100.0
# Adaptive meshing parameters
defaults['load_mesh'] = False
defaults['just_build_adaptive'] = False
defaults['adapt_tol'] = 5e-5
defaults['save_mesh'] = False
defaults['all_steps_adaptive'] = True
# defaults['mesh_file'] = 'mesh.h5'

# Far field plate rate boundary condition.
defaults['plate_rate'] = 0#(40.0 / 1.0e3) / secs_in_a_year  # 40 mm/yr

# Initial stress setup -- fed into an elastic half-space solution
# to determine initial conditions. In the future, I could numerically
# solve a Poisson problem to determine a solution that would allow
# slip variations and elastic modulus variations.
defaults['fault_slip'] = 1.0
defaults['fault_depth'] = 1.0e4
# In case I am using the "wound-up" analytic solution
defaults['recur_interval'] = 100 * secs_in_a_year
defaults['elastic_depth'] = 1.0e4
defaults['viscosity'] = 5.0e19
defaults['initial_stress'] = simple_stress

# Where to save data?
defaults['data_dir'] = 'test'

defaults['plot'] = True

# Calculate and save error
defaults['calc_error'] = True

def default_params():
    # We deepcopy in case there are important sub-objects
    p = copy.deepcopy(defaults)
    return p
