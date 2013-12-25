from dolfin import *

# Load mesh and boundary indicators from file
mesh = RectangleMesh(0, 0, 1, 1, 2, 2)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q

# Create functions
(v, q) = TestFunctions(W)
w = Function(W)
(u, p) = split(w)

# Define variational form
n = FacetNormal(mesh)
F = (0.01*inner(grad(u), grad(v)) + inner(grad(u)*u, v)
     - div(v)*p + q*div(u))*dx + dot(n, v)*ds

# No-slip boundary condition for the velocity
bc = DirichletBC(W.sub(0), Constant((1.0, 0.0)), lambda x, on_bndry: on_bndry and x[0] > 0.00000001)

# Define goal functional
M = inner(u,n)*ds

# Solve adaptively
solve(F == 0, w, bc, tol=0.01, M=M)
