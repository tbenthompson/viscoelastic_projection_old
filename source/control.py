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

def run():
    outer_start = time.clock()

    if params['load_mesh']:
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
    if not params['load_mesh']:
        print "Building Adaptive Mesh"
        vel_solver.adaptive_step(strs_solver.vel_rhs_init_adaptive())
        strs_solver = StressSolver(prob)
        vel_solver = VelocitySolver(prob)
        if params['save_mesh']:
            f = dfn.HDF5File(params['mesh_file'], 'w')
            f.write(prob.mesh, 'mesh')
        print "Done building adaptive mesh"
        if params['just_build_adaptive']:
            sys.exit()


    dt = params['delta_t']
    t = dt
    step = 0
    T = params['t_max']
    while t <= T:
        inner_start = time.clock()
        test_bc.t = t

        # strs_solver.tentative_step()

        vel_rhs = strs_solver.vel_rhs()
        vel_solver.time_step(vel_rhs)
        if params['all_steps_adaptive']:
            strs_solver = StressSolver(prob, strs_solver)
            vel_solver = VelocitySolver(prob, vel_solver)

        strs_rhs = vel_solver.strs_rhs()
        strs_solver.helmholtz_step(strs_rhs)

        vel_solver.finish_time_step()
        strs_solver.finish_time_step()
        t += dt
        step += 1
        print "t =", t
        inner_end = time.clock()
        print "Time step took: " + str(inner_end - inner_start)

    print "Done Computing"
    outer_end = time.clock()
    print "Run time: " + str(outer_end - outer_start) + " seconds"
    # Calculate the error in comparison with the analytic solution
    if params['calc_error']:
        print "Calculating Error"
        v_guess, v_exact, error, error_map = calc_error(vel_solver.cur_vel, test_bc.t)
        print "Done calculating Error"
        print "Total error: " + str(error)
        return error
