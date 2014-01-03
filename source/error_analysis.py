import dolfin as dfn
import numpy as np
import matplotlib.pyplot as pyp
from params import params
from analytic_fast import simple_velocity


def get_finer_grid(function):
    comp_width_x = 10 * params['x_points']
    comp_width_y = 10 * params['y_points']

    mesh = dfn.RectangleMesh(params['x_min'], params['y_min'],
                     params['x_max'], params['y_max'],
                     comp_width_x, comp_width_y)
    X_ = np.linspace(params['x_min'], params['x_max'], comp_width_x + 1)
    Y_ = np.linspace(params['y_min'], params['y_max'], comp_width_y + 1)
    X, Y = np.meshgrid(X_, Y_)

    linear_tris = dfn.FunctionSpace(mesh, "CG", 1)
    v_interp = dfn.interpolate(function, linear_tris)

    v_numpy = v_interp.vector()[linear_tris.dofmap().dof_to_vertex_map(mesh)].\
        array().reshape((comp_width_x + 1, comp_width_y + 1))
    return v_numpy, X, Y, mesh

def calc_error(vel, t, view=False):
    print "Converting to numpy array"
    v_guess, X, Y, mesh = get_finer_grid(vel)
    print "Done with conversion."
    v_exact = np.empty_like(v_guess)
    for i in range(v_guess.shape[0]):
        for j in range(v_guess.shape[1]):
            v_exact[i, j] = simple_velocity(
                X[i, j], Y[i, j], params['fault_depth'], t,
                params['material']['shear_modulus'],
                params['viscosity'], params['plate_rate'])
    error_map = np.abs(v_guess - v_exact)
    error = np.mean(np.abs(v_guess - v_exact)) / np.mean(v_exact)
    if params['plot']:
        view_error(v_guess, v_exact, error, error_map)
    return v_guess, v_exact, error, error_map

def view_error(v_guess, v_exact, error, error_map):
    print error
    pyp.figure(1)
    imgs = pyp.imshow(v_guess, vmin=0, vmax=np.max(v_exact))
    pyp.colorbar()
    pyp.figure(2)
    imex = pyp.imshow(v_exact, vmin=0, vmax=np.max(v_exact))
    pyp.colorbar()
    imgs.set_cmap(imex.get_cmap())
    pyp.figure(3)
    pyp.imshow(error_map, interpolation='none')
    pyp.colorbar()
    pyp.show()

