import matplotlib.pyplot as pyp
import dolfin as dfn
from params import params
from boundary_conditions import test_bc
from error_analysis import calc_error
from stress import StressSolver
from velocity import VelocitySolver
from problem import Problem
import pdb


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

prob = Problem()
strs_solver = StressSolver(prob)
vel_solver = VelocitySolver(prob)
# vel_solver.adapt_mesh(strs_solver.vel_rhs_adaptive())
# strs_solver.setup_forms()
# strs_solver.setup_forms()

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
# Calculate the error in comparison with the analytic solution
# calc_error(vel_solver.cur_vel, test_bc.t)
