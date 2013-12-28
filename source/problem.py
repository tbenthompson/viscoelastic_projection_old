import dolfin as dfn
from params import params
class Problem(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.setup_fnc_spaces()

    def setup_fnc_spaces(self):
        """
        Call me again if the mesh is modified, like after adaptive meshing.
        """
        # Linear Lagrange triangles for the velocity.
        self.v_fnc_space = dfn.FunctionSpace(self.mesh, "CG", 1)
        self.v = dfn.TrialFunction(self.v_fnc_space)
        self.vt = dfn.TestFunction(self.v_fnc_space)
        # We use piecewise constant discontinuous elements so that
        # no matrix inversion is required in order to update the stress
        self.S_fnc_space = dfn.VectorFunctionSpace(self.mesh, "DG", 0)
        self.S = dfn.TrialFunction(self.S_fnc_space)
        self.St = dfn.TestFunction(self.S_fnc_space)

    def update_mesh(self, new_mesh):
        self.mesh = new_mesh
        self.setup_fnc_spaces()
