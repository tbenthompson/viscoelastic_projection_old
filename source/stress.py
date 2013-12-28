import numpy as np
import dolfin as dfn
import scipy.sparse as sparse
from params import params
from analytic_fast import simple_stress

class InitialStress(dfn.Expression):
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
    def __init__(self, prob):
        self.prob = prob
        self.file = dfn.File("../data/stress.pvd")
        self.inv_eta = InvViscosity(cell=dfn.triangle)
        self.inv_eta.set_params(params['elastic_depth'], params['viscosity'])
        self.dt = dfn.Constant(params['delta_t'])
        self.mu = params['material']['shear_modulus']
        self.init_strs = InitialStress(cell=dfn.triangle)
        self.init_strs.set_params(params['fault_slip'],
                                  params['fault_depth'],
                                  params['recur_interval'],
                                  params['material']['shear_modulus'],
                                  params['viscosity'],
                                  params['plate_rate'])
        self.setup_forms()

    def setup_forms(self):
        prob = self.prob
        self.old_strs = dfn.interpolate(self.init_strs, prob.S_fnc_space)
        self.cur_strs = dfn.Function(prob.S_fnc_space)

        self.a = dfn.inner(prob.S, prob.St) * dfn.dx
        self.l_visc = dfn.inner(self._tentative_update(prob.S), prob.St) * dfn.dx

        self.l_div_strs = self._div_strs(prob.S)
        self.l_div_strs_initial = self._div_strs(self.init_strs)

        self.A = dfn.assemble(self.a)
        diag = dfn.as_backend_type(self.A).mat().getDiagonal()
        diag.reciprocal()
        dfn.as_backend_type(self.A).mat().setDiagonal(diag)
        self.A_inv = self.A
        self.L_visc = dfn.assemble(self.l_visc)
        self.L_div_strs = dfn.assemble(self.l_div_strs)

    def _tentative_update(self, S):
        initial_strs = S
        f1 = -self.dt * self.mu * self.inv_eta * S
        return initial_strs + f1

    def _div_strs(self, S):
        prob = self.prob
        term = (1 / (self.mu * self.dt)) * \
            dfn.div(self._tentative_update(S)) * prob.vt * dfn.dx
        return term

    def vel_rhs_adaptive(self):
        return self.l_div_strs_initial

    def vel_rhs(self):
        return self.L_div_strs * self.old_strs.vector()

    def time_step(self, rhs):
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
