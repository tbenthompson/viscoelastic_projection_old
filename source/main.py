import scitools.BoxField
import matplotlib.pyplot as pyp
from dolfin import *
from params import params
import numpy as np
from analytic_fast import simple_velocity, simple_stress
import pdb


def _DEBUG():
    pdb.set_trace()


class InvViscosity(Expression):

    def set_params(self, D, eta):
        self.D = D
        self.eta = eta

    def eval(self, value, x):
        value[0] = 0.0
        if x[1] > self.D:
            value[0] = 1.0 / self.eta


class TestBC(Expression):

    def set_params(self, D, recur_interval,
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



def fault_boundary(x, on_boundary):
    return on_boundary and x[0] < params['x_min'] + 0.1


def plate_boundary(x, on_boundary):
    return on_boundary and x[0] > params['x_max'] - 0.1


def mantle_boundary(x, on_boundary):
    return on_boundary and x[1] > params['y_max'] - 0.1


def testing_boundary(x, on_bdry):
    return plate_boundary(x, on_bdry) or \
        mantle_boundary(x, on_bdry)   or \
        fault_boundary(x, on_bdry)


def get_normal_bcs(fnc_space):
    fault = DirichletBC(fnc_space,
                        Constant(0.0),
                        fault_boundary)
    plate = DirichletBC(fnc_space,
                        Constant(params['plate_rate']),
                        plate_boundary)
    mantle = DirichletBC(fnc_space,
                         Constant(0.0),
                         mantle_boundary)
    return [fault, plate, mantle]


def get_test_bcs(fnc_space, bc):
    testing = DirichletBC(fnc_space,
                          bc,
                          testing_boundary)
    return [testing]


class InitialStress(Expression):
    def set_params(self, s, D, recur, mu, viscosity, plate_rate):
        self.recur = recur
        self.mu = mu
        self.D = D
        self.s = s
        self.viscosity = viscosity
        self.plate_rate = plate_rate

    def eval(self, value, x):
        Szx, Szy = self._eval(x)
        value[0] = Szx
        value[1] = Szy

    def _eval(self, x):
        return simple_stress(x[0], x[1], self.s, self.D, self.mu)

    def value_shape(self):
        return (2,)
def nparray_from_fenics(mesh, function):
    X_ = np.linspace(params['x_min'], params['x_max'], nx + 1)
    Y_ = np.linspace(params['y_min'], params['y_max'], ny + 1)
    X, Y = np.meshgrid(X_, Y_)
    linear_tris = FunctionSpace(mesh, "CG", 1)
    v_interp = interpolate(function, linear_tris)

    v_numpy = v_interp.vector()[linear_tris.dofmap().dof_to_vertex_map(mesh)].\
        array().reshape((ny + 1, nx + 1))
    return v_numpy, X, Y

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False
# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True
# Allow approximating values for points that may be generated outside
# of domain (because of numerical inaccuracies)
parameters["allow_extrapolation"] = True
# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
# set_log_level(1)

# Set parameter values
dt = params['delta_t']
T = params['t_max']
mu = Constant(params['material']['shear_modulus'])
inv_eta = InvViscosity(cell=triangle)
inv_eta.set_params(params['elastic_depth'], params['viscosity'])
# Testing Boundary Conditions (Velocity at Mantle and Plate Boundaries)
test_bc = TestBC()
test_bc.set_params(params['fault_depth'],
                 params['recur_interval'],
                 params['material']['shear_modulus'],
                 params['viscosity'],
                 params['plate_rate'])
test_bc.t = dt
# Initial stress conditions
initial_stress = InitialStress(cell=triangle)
initial_stress.set_params(params['fault_slip'],
                               params['fault_depth'],
                               params['recur_interval'],
                               params['material']['shear_modulus'],
                               params['viscosity'],
                               params['plate_rate'])
k = Constant(dt)
f = Constant((0, 0))
# Create files for storing solution
sfile = File("../data/stress.pvd")
vfile = File("../data/velocity.pvd")

# Create mesh
nx = params['x_points']
ny = params['y_points']
mesh = RectangleMesh(params['x_min'], params['y_min'],
                     params['x_max'], params['y_max'],
                     nx, ny)
# TOL = 1e-6
# REFINE_RATIO=0.10
new_mesh = mesh
# for level in range(500):
    # Determine the required mesh.
v_fnc_space = FunctionSpace(new_mesh, "CG", 1)
v = TrialFunction(v_fnc_space)
vt = TestFunction(v_fnc_space)
# bcs = get_normal_bcs(v_fnc_space)
bcs = get_test_bcs(v_fnc_space, test_bc)
v1 = Function(v_fnc_space)
a2 = inner(grad(v), grad(vt)) * dx
A2 = assemble(a2)
l2 = (1 / (mu * k)) * div((-k * mu * inv_eta) * initial_stress) * vt * dx
solve(a2 == l2, v1, bcs, tol=1e-6, M=v1*dx)
new_mesh = mesh.leaf_node()
    # b2 = assemble(l2)
    # [bc.apply(A2, b2) for bc in bcs]
    # solve(A2, v1.vector(), b2)
    # h = np.array([c.diameter() for c in cells(new_mesh)])
    # K = np.array([c.volume() for c in cells(new_mesh)])
    # R = np.array([np.mean([b2.array()[dof] for dof in v_fnc_space.dofmap().cell_dofs(c.index())])
    #               for c in cells(new_mesh)])
    # gamma = h*R*np.sqrt(K)

    # # Compute error estimate
    # E = sum([g*g for g in gamma])
    # E = sqrt(MPI.sum(E))
    # print "Level %d: E = %g (TOL = %g)" % (level, E, TOL)

    # # Check convergence
    # if E < TOL:
    #     info("Success, solution converged after %d iterations", level)
    #     break

    # # Mark cells for refinement
    # cell_markers = MeshFunction("bool", new_mesh, new_mesh.topology().dim())
    # gamma_0 = sorted(gamma, reverse=True)[int(len(gamma)*REFINE_RATIO)]
    # for c in cells(new_mesh):
    #     cell_markers[c] = gamma[c.index()] > gamma_0

    # # Refine mesh
    # new_mesh = refine(new_mesh, cell_markers)

    # Plot mesh
plot(new_mesh)


#Define function spaces
S_fnc_space = VectorFunctionSpace(new_mesh, "DG", 0)
v_fnc_space = FunctionSpace(new_mesh, "CG", 1)

#Define test and trial functions
S = TrialFunction(S_fnc_space)
St = TestFunction(S_fnc_space)
v = TrialFunction(v_fnc_space)
vt = TestFunction(v_fnc_space)

# Boundary Conditions
# bcs = get_normal_bcs(v_fnc_space)
bcs = get_test_bcs(v_fnc_space, test_bc)

# New velocity
v0 = Function(v_fnc_space)
v1 = Function(v_fnc_space)

# Old stress
S0 = interpolate(initial_stress, S_fnc_space)
# New stress
S1 = Function(S_fnc_space)
Szx, Szy = S1.split()

# Velocity update
a2 = inner(grad(v), grad(vt)) * dx
l2 = (1 / (mu * k)) * div((-k * mu * inv_eta) * S) * vt * dx

# Helmholtz decomposition stress update
a3 = inner(S, St) * dx
l3_1 = inner((1 - k * mu * inv_eta) * S, St) * dx
l3_2 = k * mu * inner(grad(v), St) * dx

# Assemble lhs matrices
A2 = assemble(a2)
L2 = assemble(l2)
A3 = assemble(a3)
_DEBUG()
L3_1 = assemble(l3_1)
L3_2 = assemble(l3_2)

# Time-step
t = dt
i = 1
while t < T + DOLFIN_EPS:
    test_bc.t = t

    # Velocity correction
    begin("Computing velocity correction")
    b2 = L2 * S0.vector()
    [bc.apply(A2, b2) for bc in bcs]
    solve(A2, v1.vector(), b2, "cg", prec)
    end()

    # Velocity correction
    begin("Computing stress correction")
    b3 = L3_1 * S0.vector() + L3_2 * v1.vector()
    solve(A3, S1.vector(), b3, "cg", prec)
    end()

    # Plot
    plot(v1)

    # Save to file
    sfile << S1
    vfile << v1

    # Move to next time step
    S0.assign(S1)
    v0.assign(v1)
    t += dt
    i += 1
    print "t =", t
print "Done Computing"

def calc_error(which_mesh):
    print "Converting to numpy array"
    v_guess, X, Y = nparray_from_fenics(which_mesh, v1)
    print "Done with conversion."
    v_exact = np.empty_like(v_guess)
    for i in range(v_guess.shape[0]):
        for j in range(v_guess.shape[1]):
            v_exact[i, j] = simple_velocity(
                X[i, j], Y[i, j], params['fault_depth'], test_bc.t,
                params['material']['shear_modulus'],
                params['viscosity'], params['plate_rate'])
    error = np.mean(np.abs(v_guess - v_exact)) / np.mean(v_exact)
    print error
    pyp.figure(1)
    imgs = pyp.imshow(v_guess, vmin=0, vmax=np.max(v_exact))
    pyp.colorbar()
    pyp.figure(2)
    imex = pyp.imshow(v_exact, vmin=0, vmax=np.max(v_exact))
    pyp.colorbar()
    imgs.set_cmap(imex.get_cmap())
    pyp.show()
# Iterate over solution and calculate error
calc_error(mesh.root_node())
