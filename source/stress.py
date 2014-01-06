import numpy as np
import dolfin as dfn
import scipy.sparse as sparse
from params import params
from analytic_fast import simple_stress


class InitialStress(dfn.Expression):
    def eval(self, value, x):
        Szx, Szy = params['initial_stress'](x[0], x[1])
        value[0] = Szx
        value[1] = Szy

    def value_shape(self):
        return (2,)


class InvViscosity(dfn.Expression):

    """
    Inverse of viscosity (1/eta)
    """

    def set_params(self, D, eta):
        self.D = D
        self.eta = eta

    def eval(self, value, x):
        value[0] = 0.0
        if x[1] > self.D:
            value[0] = 1.0 / self.eta


class StressSolver(object):

    def __init__(self, prob, old_solver=None):
        self.prob = prob
        self.file = dfn.File("../data/stress.pvd")
        self.ode_method = self._tent_rk4
        self.inv_eta = InvViscosity(cell=dfn.triangle)
        self.inv_eta.set_params(params['elastic_depth'], params['viscosity'])
        self.dt = dfn.Constant(params['delta_t'])
        self.mu = dfn.Constant(params['material']['shear_modulus'])
        self.init_cond = InitialStress(cell=dfn.triangle)
        self.setup_forms(old_solver)

    def setup_forms(self, old_solver):
        prob = self.prob
        if old_solver is not None:
            self.cur_strs = dfn.interpolate(old_solver.cur_strs,
                                                      prob.S_fnc_space)
            self.old_strs = dfn.interpolate(old_solver.old_strs,
                                                      prob.S_fnc_space)
        else:
            self.old_strs = dfn.interpolate(self.init_cond,
                                            self.prob.S_fnc_space)
            self.cur_strs = dfn.Function(self.prob.S_fnc_space)

        self.a = dfn.inner(prob.S, prob.St) * dfn.dx
        self.l_visc = dfn.inner(self.ode_method(prob.S), prob.St) * dfn.dx

        self.l_div_strs = self._div_strs(prob.S)
        self.l_div_strs_initial = self._div_strs(self.init_cond)
        self.l_div_strs_adaptive = self._div_strs(self.old_strs)

        self.A = dfn.assemble(self.a)
        diag = dfn.as_backend_type(self.A).mat().getDiagonal()
        diag.reciprocal()
        dfn.as_backend_type(self.A).mat().setDiagonal(diag)
        self.A_inv = self.A
        self.L_visc = dfn.assemble(self.l_visc)
        self.L_div_strs = dfn.assemble(self.l_div_strs)

    def _div_strs(self, S):
        prob = self.prob
        term = (1 / (self.mu * self.dt)) * \
            dfn.div(self.ode_method(S)) * prob.vt * dfn.dx
        return term

    def vel_rhs_init_adaptive(self):
        return self.l_div_strs_initial

    def vel_rhs_adaptive(self):
        return self.l_div_strs_adaptive

    def vel_rhs_simple(self):
        return self.L_div_strs * self.old_strs.vector()

    def vel_rhs(self):
        if params['all_steps_adaptive']:
            return self.vel_rhs_adaptive()
        else:
            return self.vel_rhs_simple()

    # def tentative_step(self):
    #     print("Computing tentative stress")
    #     b = self.L_visc * self.old_strs.vector()
    #     update = self.A_inv * b
    #     self.cur_strs.vector()[:] = update[:]
    #     print("Done computing tentative_stress")

    # def helmholtz_step(self, rhs):
    #     print("Computing stress correction")
    #     b = rhs
    #     update = self.cur_strs.vector() + self.A_inv * b
    #     self.cur_strs.vector()[:] = update[:]
    #     print("Done computing Stress Correction")

    def helmholtz_step(self, rhs):
        print("Computing stress correction")
        b = self.L_visc * self.old_strs.vector() + rhs
        update = self.A_inv * b
        self.cur_strs.vector()[:] = update[:]
        print("Done computing Stress Correction")

    def finish_time_step(self):
        # self.save()
        self.old_strs.assign(self.cur_strs)

    def save(self):
        self.file << self.cur_strs

    def f(self, S):
        return -self.mu * self.inv_eta * S

    def _tent_euler(self, S):
        f1 = self.f(S)
        result = S + self.dt * f1
        return result

    def _tent_heun(self, S):
        f1 = self.f(S)
        f2 = self.f(S + self.dt * f1)
        result = S + (0.5 * self.dt * (f1 + f2))
        return result

    def _tent_rk4(self, S):
        f1 = self.f(S)
        f2 = self.f(S + (0.5 * self.dt * f1))
        f3 = self.f(S + (0.5 * self.dt * f2))
        f4 = self.f(S + (1.0 * self.dt * f3))
        result = S + \
            ((1.0 / 6.0) * self.dt * f1) + \
            ((1.0 / 3.0) * self.dt * f2) + \
            ((1.0 / 3.0) * self.dt * f3) + \
            ((1.0 / 6.0) * self.dt * f4)
        return result
