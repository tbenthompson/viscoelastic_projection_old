import scitools.BoxField
import matplotlib.pyplot as pyp
from dolfin import *
from params import params
import numpy as np
from analytic_fast import simple_velocity, simple_stress

import pdb
def _DEBUG():
    pdb.set_trace()

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False

# Create mesh
nx = params['x_points']
ny = params['y_points']
mesh = RectangleMesh(params['x_min'], params['y_min'],
                     params['x_max'], params['y_max'],
                     nx, ny)

# Define function spaces
S_fnc_space = VectorFunctionSpace(mesh, "CG", 2)
v_fnc_space = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
S = TrialFunction(S_fnc_space)
v = TrialFunction(v_fnc_space)
St = TestFunction(S_fnc_space)
vt = TestFunction(v_fnc_space)

# Set parameter values
dt = params['delta_t']
T = params['t_max']

# material parameters
mu = Constant(params['material']['shear_modulus'])


class InvViscosity(Expression):
    def __init__(self, D, eta):
        self.D = D
        self.eta = eta

    def eval(self, value, x):
        value[0] = 0.0
        if x[1] > self.D:
            value[0] = 1.0 / self.eta
inv_eta = InvViscosity(params['elastic_depth'], params['viscosity'])

# Define boundary conditions
tol = 1e-10


class TestBC(Expression):
    def __init__(self, D, recur_interval,
                 shear_modulus, viscosity, plate_rate):
        self.D = D
        self.recur_interval = recur_interval
        self.shear_modulus = shear_modulus
        self.viscosity = viscosity
        self.plate_rate = plate_rate
        self.t = 0

    def eval(self, value, x):
        value[0] = simple_velocity(x[0], x[1],
                                        self.D,
                                        self.t,
                                        self.shear_modulus,
                                        self.viscosity,
                                        self.plate_rate)
        # value[0] = velocity_dimensional(x[0], x[1],
        #                                 self.D,
        #                                 self.t,
        #                                 self.recur_interval,
        #                                 self.shear_modulus,
        #                                 self.viscosity,
        #                                 self.plate_rate,
        #                                 past_events=50)
test_bc = TestBC(params['fault_depth'],
                 params['recur_interval'],
                 params['material']['shear_modulus'],
                 params['viscosity'],
                 params['plate_rate'])
def fault_boundary(x, on_boundary):
    return on_boundary and x[0] < params['x_min'] + tol
def plate_boundary(x, on_boundary):
    return on_boundary and x[0] > params['x_max'] - tol
def mantle_boundary(x, on_boundary):
    return on_boundary and x[1] > params['y_max'] - tol
def testing_boundary(x, on_bdry):
    return plate_boundary(x, on_bdry) or \
        mantle_boundary(x, on_bdry) or \
        fault_boundary(x, on_bdry)
def get_normal_bcs():
    fault = DirichletBC(v_fnc_space,
                        Constant(0.0),
                        fault_boundary)
    plate = DirichletBC(v_fnc_space,
                        Constant(params['plate_rate']),
                        plate_boundary)
    mantle = DirichletBC(v_fnc_space,
                         Constant(0.0),
                         mantle_boundary)
    return [fault, plate, mantle]
def get_test_bcs():
    testing = DirichletBC(v_fnc_space,
                          test_bc,
                          testing_boundary)
    return [testing]
# bcs = get_normal_bcs()
bcs = get_test_bcs()

# Define Initial Conditions
class InitialStress(Expression):
    def __init__(self, s, D, recur, mu, viscosity, plate_rate):
        self.recur = recur
        self.mu = mu
        self.D = D
        self.s = s
        self.viscosity = viscosity
        self.plate_rate = plate_rate
    def eval(self, value, x):
        Szx, Szy = simple_stress(x[0], x[1], self.s, self.D, self.mu)
        # Szx, Szy = stress_dimensional(x[0], x[1], self.D, 0.0, self.recur, self.mu,
        #                               self.viscosity, self.plate_rate)
        value[0] = Szx
        value[1] = Szy
    def value_shape(self):
        return (2,)
initial_stress = InitialStress(params['fault_slip'],
                               params['fault_depth'],
                               params['recur_interval'],
                               params['material']['shear_modulus'],
                               params['viscosity'],
                               params['plate_rate'])


# Old stress
S0 = interpolate(initial_stress, S_fnc_space)
# New stress
S1 = Function(S_fnc_space)
S2 = Function(S_fnc_space)
Szx, Szy = S1.split()
# New velocity
v0 = Function(v_fnc_space)
v1 = Function(v_fnc_space)


# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# a33 = v * vt * dx
# L33 = div(S0) * vt * dx
# A33 = assemble(a33)
# b33 = assemble(L33)
# solve(A33, v0.vector(), b33)
# _DEBUG()

# Tentative stress step
A1 = inner(S - S0, St) * dx + \
    k * mu * inv_eta * inner(S, St) * dx
    # inner(Constant((0,0)), St) * dx
a1 = lhs(A1)
L1 = rhs(A1)

# Velocity update
a2 = inner(grad(v), grad(vt)) * dx
# L2 = Constant(0) * vt * dx
L2 = (1 / (mu * k)) * div(S1) * vt * dx

# Helmholtz decomposition stress update
a3 = inner(S, St) * dx
L3 = inner(S1, St) * dx + k * mu * inner(grad(v1), St) * dx

# Assemble lhs matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Create files for storing solution
sfile = File("../data/stress.pvd")
vfile = File("../data/velocity.pvd")

# Time-step
t = dt

def solve_tentative_stress():
    begin("Computing tentative stress")
    b1 = assemble(L1)
    solve(A1, S1.vector(), b1, "cg", prec)
    # S0.assign(S1)
    # solve(A33, v0.vector(), b33)
    # _DEBUG()
    end()

def solve_velocity():
    begin("Computing velocity correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcs]
    solve(A2, v1.vector(), b2, "cg", prec)
    end()

def solve_stress_helmholtz():
    begin("Computing stress correction")
    b3 = assemble(L3)
    solve(A3, S2.vector(), b3, "cg", prec)
    end()

while t < T + DOLFIN_EPS:
    test_bc.t = t
    # Compute tentative stress step
    solve_tentative_stress()

    # Velocity correction
    solve_velocity()

    # Velocity correction
    solve_stress_helmholtz()

    # Plot solution
    # plot(Szx)
    # plot(v1)

    # Save to file
    sfile << S2
    vfile << v1

    # Move to next time step
    S0.assign(S2)
    v0.assign(v1)
    t += dt
    print "t =", t

# Iterate over solution and calculate error
X_ = np.linspace(params['x_min'], params['x_max'], nx + 1)
Y_ = np.linspace(params['y_min'], params['y_max'], ny + 1)
X, Y = np.meshgrid(X_, Y_)
v_guess = v1.vector()[v_fnc_space.dofmap().dof_to_vertex_map(mesh)].\
            array().reshape((ny + 1, nx + 1))
v_exact = np.empty_like(v_guess)
for i in range(v_guess.shape[0]):
    for j in range(v_guess.shape[1]):
        v_exact[i, j] = simple_velocity(X[i, j], Y[i, j], params['fault_depth'], test_bc.t,
                               params['material']['shear_modulus'],
                               params['viscosity'], params['plate_rate'])
# v_exact = velocity_dimensional(X, Y, params['fault_depth'], test_bc.t,
#                                0.0, params['material']['shear_modulus'],
#                                params['viscosity'], params['plate_rate'])
error = np.mean(np.abs(v_guess - v_exact)) / np.mean(v_exact)
print error
# pyp.figure(1)
# pyp.imshow(v_guess)
# pyp.colorbar()
# pyp.figure(2)
# pyp.imshow(v_exact)
# pyp.colorbar()
# pyp.show()
import sys
sys.exit()


# Hold plot
# interactive()
