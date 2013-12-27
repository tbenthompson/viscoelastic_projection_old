import dolfin as dfn
from boundary_conditions import get_test_bcs, test_bc
from params import params
from stress import InitialStress, InvViscosity
class VelocitySolver(object):
    def __init__(self, prob):
        self.prob = prob
        self.file = dfn.File("../data/velocity.pvd")
        self.mu = params['material']['shear_modulus']
        self.dt = dfn.Constant(params['delta_t'])
        # Initial stress conditions
        self.inv_eta = InvViscosity(cell=dfn.triangle)
        self.inv_eta.set_params(params['elastic_depth'], params['viscosity'])
        self.setup_forms()

    def setup_forms(self):
        prob = self.prob
        self.old_vel = dfn.Function(prob.v_fnc_space)
        self.cur_vel = dfn.Function(prob.v_fnc_space)
        self.a = dfn.inner(dfn.grad(prob.v), dfn.grad(prob.vt)) * dfn.dx
        # self.l = (1 / (mu * k)) * div((1 - k * mu * inv_eta) * S) * vt * dx
        self.bcs = get_test_bcs(prob.v_fnc_space, test_bc)
        # Assemble matrices from variational forms. Ax = Ls
        self.A = dfn.assemble(self.a)
        # self.L = dfn.assemble(self.l)

    def adapt_mesh(self):
        mu, dt, inv_eta = self.mu, self.dt, self.inv_eta
        rhs = (1 / (mu * dt)) * dfn.div((1 - dt * mu * inv_eta) * self.init_strs) * \
            self.vt * dfn.dx
        dfn.solve(self.a == rhs, self.cur_vel, self.bcs, tol=1e-2, M=self.cur_vel*dfn.dx)
        self.mesh = self.mesh.leaf_node()
        dfn.plot(self.mesh)
        dfn.interactive()
        self.setup_forms()

    def time_step(self, rhs):
        dfn.begin("Computing velocity correction")
        [bc.apply(self.A, rhs) for bc in self.bcs]
        dfn.solve(self.A, self.cur_vel, rhs, "cg", "amg")
        dfn.end()

    def finish_time_step(self):
        self.save()
        self.old_vel.assign(self.cur_vel)

    def save(self):
        self.file << self.cur_vel

