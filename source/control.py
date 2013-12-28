import sys
import pdb
import time
import matplotlib.pyplot as pyp
import dolfin as dfn
from params import params
from boundary_conditions import test_bc
from error_analysis import calc_error
from stress import StressSolver
from velocity import VelocitySolver
from problem import Problem


def _DEBUG():
    pdb.set_trace()

# Set some FEnICs parameters for control
# Print log messages only from the root process in parallel
dfn.parameters["std_out_all_processes"] = False
# Use compiler optimizations
dfn.parameters["form_compiler"]["cpp_optimize"] = True
# Allow approximating values for points that may be generated outside
# of domain (because of numerical inaccuracies)
dfn.parameters["allow_extrapolation"] = True
start = time.clock()

if params['load']:
    f = dfn.HDF5File(params['mesh_file'], 'r')
    m = dfn.Mesh()
    f.read(m, 'mesh')
else:
    m = dfn.RectangleMesh(params['x_min'], params['y_min'],
                              params['x_max'], params['y_max'],
                              params['x_points'], params['y_points'])
prob = Problem(m)
strs_solver = StressSolver(prob)
vel_solver = VelocitySolver(prob)
if params['load'] is False:
    vel_solver.adapt_mesh(strs_solver.vel_rhs_adaptive())
    f = dfn.HDF5File(params['mesh_file'], 'w')
    f.write(prob.mesh, 'mesh')
    sys.exit()

dt = params['delta_t']
t = dt
step = 0
T = params['t_max']
while t <= T:
    test_bc.t = t

    vel_rhs = strs_solver.vel_rhs()
    vel_solver.time_step(vel_rhs)
    strs_rhs = vel_solver.strs_rhs()
    strs_solver.time_step(strs_rhs)

    vel_solver.finish_time_step()
    strs_solver.finish_time_step()
    t += dt
    step += 1
    print "t =", t

print "Done Computing"
end = time.clock()
print "Run time: " + str(end - start) + " seconds"
# Calculate the error in comparison with the analytic solution
calc_error(vel_solver.cur_vel, test_bc.t)
