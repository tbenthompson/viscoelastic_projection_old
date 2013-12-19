"""This demo program solves the incompressible Navier-Stokes equations
on an L-shaped domain using Chorin's splitting method."""

# Copyright (C) 2010-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Mikael Mortensen 2011
#
# First added:  2010-08-30
# Last changed: 2011-06-30

# Begin demo

from dolfin import *
from core.debug import _DEBUG
from parameters.proj import params

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
mesh = RectangleMesh(params.x_min, params.y_min,
                     params.x_max, params.y_max,
                     params.x_points, params.y_points)

# Define function spaces (P2-P1)
S_fnc_space = VectorFunctionSpace(mesh, "Lagrange", 2)
v_fnc_space = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
S = TrialFunction(S_fnc_space)
v = TrialFunction(v_fnc_space)
St = TestFunction(S_fnc_space)
vt = TestFunction(v_fnc_space)

# Set parameter values
dt = params.delta_t
T = params.t_max

#material parameters
mu = Constant(params.material.shear_modulus)
class Viscosity(Expression):
    def __init__(self, D, eta):
        self.D = D
        self.eta = eta
    def eval(self, values, x):
        value[0] = 0.0
        if x[1] > self.D:
            value[0] = self.eta
eta = Viscosity(params.elastic_depth, params.viscosity)

# Define boundary conditions
tol = 1e-10
def fault_boundary(x, on_boundary):
    return on_boundary and x[0] < tol
def plate_boundary(x, on_boundary):
    return on_boundary and x[0] > params.x_max - tol
def mantle_boundary(x, on_boundary):
    return on_boundary and x[1] > params.y_max - tol
fault = DirichletBC(v_fnc_space,
                     Constant(0.0),
                      fault_boundary)
plate = DirichletBC(v_fnc_space,
                      Constant(params.plate_rate),
                      plate_boundary)
mantle = DirichletBC(v_fnc_space,
                     Constant(0.0),
                      mantle_boundary)
bcs = [fault, plate, mantle]

# Old stress
S0 = Function(S_fnc_space)
# Tentative stress
S1 = Function(S_fnc_space)
# New velocity
v = Function(v_fnc_space)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative stress step
F1 = (1 / k) * inner(S - S0, St) * dx + \
    (mu / eta) * inner(S, St) * dx
a1 = lhs(F1)
L1 = rhs(F1)

# Velocity update
# a2 = inner(grad(p), grad(q))*dx
# L2 = -(1 / k)*div(u1)*q*dx
#
# # Helmholtz decomposition stress update
# a3 = inner(u, v)*dx
# L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
# A2 = assemble(a2)
# A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
_DEBUG()

# Create files for storing solution
sfile = File("results/stress.pvd")
vfile = File("results/velocity.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:
    # Compute tentative stress step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "cg", prec)
    end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")
    end()

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True)

    # Save to file
    sfile << u1
    vfile << p1

    # Move to next time step
    u0.assign(u1)
    t += dt
    print "t =", t

# Hold plot
interactive()
