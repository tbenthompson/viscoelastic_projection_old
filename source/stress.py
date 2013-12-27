import numpy as np
import dolfin as dfn
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
        self.sfile = dfn.File("../data/stress.pvd")
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
        self.l_visc = dfn.inner((1 - self.dt * self.mu * self.inv_eta) * prob.S, prob.St) * dfn.dx
        self.l_elast = l3_2 = self.dt * self.mu * dfn.inner(dfn.grad(prob.v), prob.St) * dfn.dx

        self.l_div_strs = (1 / (self.mu * self.dt)) * \
            dfn.div((1 - self.dt * self.mu * self.inv_eta) * prob.S) * prob.vt * dfn.dx

        self.A = dfn.assemble(self.a)
        self.A_inv = np.diag(1.0 / np.diagonal(self.A.array()))
        self.L_visc = dfn.assemble(self.l_visc)
        self.L_elast = dfn.assemble(self.l_elast)
        self.L_div_strs = dfn.assemble(self.l_div_strs)

    def vel_rhs(self):
        return self.L_div_strs * self.old_strs

    def time_step(self):
        dfn.begin("Computing stress correction")
        b3 = L3_1 * self.old_strs.vector()#FIGURE OUT! + L3_2 * 1self.vector()
        update = A3a_inv.dot(b3.array())
        S1.vector()[:] = update[:]
        dfn.end()

    def finish_time_step(self):
        self.save()
        self.old_strs.assign(self.cur_strs)

    def save(self):
        self.file << self.cur_strs
