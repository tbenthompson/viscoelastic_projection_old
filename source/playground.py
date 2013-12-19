from dolfin import *
mesh = UnitSquareMesh(2, 2)
coords = mesh.coordinates()

V = FunctionSpace(mesh, 'CG', 2)
vertex_to_dof_map = V.dofmap().vertex_to_dof_map(mesh)
dof_to_vertex_map = V.dofmap().dof_to_vertex_map(mesh)

u = Function(V)
x = u.vector()
dofs_at_vertices = x[dof_to_vertex_map]

# let's iterate over vertices
for v in vertices(mesh):
    print 'vertex index', v.index()
    print 'at point', v.point().str()
    print 'at coordinates', coords[v.index()]
    print 'dof', dofs_at_vertices[v.index()]
