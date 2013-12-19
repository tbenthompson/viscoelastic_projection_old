import scitools.BoxField
import matplotlib.pyplot as pyp
from dolfin import *
from params import params
from analytic import elastic_stress

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

# Define function spaces (P2-P1)
S_fnc_space = VectorFunctionSpace(mesh, "CG", 2)
v_fnc_space = FunctionSpace(mesh, "CG", 2)

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


def fault_boundary(x, on_boundary):
    return on_boundary and x[0] < params['x_min'] + tol
def plate_boundary(x, on_boundary):
    return on_boundary and x[0] > params['x_max'] - tol
def mantle_boundary(x, on_boundary):
    return on_boundary and x[1] > params['y_max'] - tol
fault = DirichletBC(v_fnc_space,
                    Constant(0.0),
                    fault_boundary)
plate = DirichletBC(v_fnc_space,
                    Constant(params['plate_rate']),
                    plate_boundary)
mantle = DirichletBC(v_fnc_space,
                     Constant(0.0),
                     mantle_boundary)
bcs = [fault, plate, mantle]

# Define Initial Conditions
class InitialStress(Expression):
    def __init__(self, s, D, mu):
        self.mu = mu
        self.D = D
        self.s = s
    def eval(self, value, x):
        Szx, Szy = elastic_stress(x[0], x[1], self.s, self.D, self.mu)
        value[0] = Szx
        value[1] = Szy
    def value_shape(self):
        return (2,)
initial_stress = InitialStress(params['fault_slip'],
                               params['fault_depth'],
                               params['material']['shear_modulus'])


# Old stress
S0 = interpolate(initial_stress, S_fnc_space)
# New stress
S1 = Function(S_fnc_space)
Szx, Szy = S1.split()
# New velocity
v1 = Function(v_fnc_space)


# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative stress step
F1 = (1 / k) * inner(S - S0, St) * dx + \
    (mu * inv_eta) * inner(S, St) * dx
a1 = lhs(F1)
L1 = rhs(F1)

# Velocity update
a2 = inner(grad(v), grad(vt)) * dx
L2 = (1 / (mu * k)) * div(S1) * vt * dx

# Helmholtz decomposition stress update
a3 = inner(S, St) * dx
L3 = inner(S1, St) * dx + k * mu * inner(grad(v1), St) * dx

# Assemble matrices
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
    solve(A3, S1.vector(), b3, "cg", prec)
    end()

while t < T + DOLFIN_EPS:
    # Compute tentative stress step
    solve_tentative_stress()

    # Velocity correction
    solve_velocity()

    # Velocity correction
    solve_stress_helmholtz()

    # Plot solution
    plot(Szx)
    plot(v1)

    # Save to file
    sfile << S1
    vfile << v1

    # Move to next time step
    S0.assign(S1)
    t += dt
    print "t =", t

# Hold plot
# interactive()
