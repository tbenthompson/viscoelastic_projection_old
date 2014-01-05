import dolfin as dfn
from boundary_conditions import get_normal_bcs, get_test_bcs, test_bc
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

        # self.bcs = get_normal_bcs(prob.v_fnc_space, test_bc)
        self.bcs = get_test_bcs(prob.v_fnc_space, test_bc)
        # Assemble matrices from variational forms. Ax = Ls
        self.A = dfn.assemble(self.a)

        # For the gradient term in the stress update
        self.l_elastic = self.dt * self.mu * \
            dfn.inner(dfn.grad(prob.v), prob.St) * dfn.dx
        self.L_elastic = dfn.assemble(self.l_elastic)

    def adapt_mesh(self, rhs):
        goal = self.cur_vel * dfn.dx
        var_prob = dfn.LinearVariationalProblem(
            self.a, rhs, self.cur_vel, self.bcs)
        var_solve = self.get_var_solver(var_prob, goal)
        var_solve.solve(params['adapt_tol'])
        self.prob.update_mesh(self.prob.mesh.leaf_node())
        # dfn.plot(self.prob.mesh)
        # dfn.interactive()

    def get_var_solver(self, var_prob, goal):
        var_solve = dfn.AdaptiveLinearVariationalSolver(var_prob, goal)
        p = var_solve.parameters
        p['linear_variational_solver']['linear_solver'] = 'cg'
        p['linear_variational_solver']['preconditioner'] = 'amg'
        p['error_control']['dual_variational_solver']['linear_solver'] = 'cg'
        p['error_control']['dual_variational_solver']['preconditioner'] = 'amg'
        return var_solve

    def strs_rhs(self):
        return self.L_elastic * self.cur_vel.vector()

    def time_step(self, rhs):
        dfn.begin("Computing velocity correction")
        if params['all_steps_adaptive']:
            goal = self.cur_vel * dfn.dx
            var_prob = dfn.LinearVariationalProblem(
                self.a, rhs, self.cur_vel, self.bcs)
            var_solve = self.get_var_solver(var_prob, goal)
            var_solve.solve(params['adapt_tol'])
        else:
            [bc.apply(self.A, rhs) for bc in self.bcs]
            dfn.solve(self.A, self.cur_vel.vector(), rhs, "cg", "amg")
        dfn.end()

    def finish_time_step(self):
        # self.save()
        self.old_vel.assign(self.cur_vel)

    def save(self):
        self.file << self.cur_vel
