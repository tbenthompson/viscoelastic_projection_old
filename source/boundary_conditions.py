import dolfin as dfn
from analytic_fast import simple_velocity
from params import params

class TestBC(dfn.Expression):

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

def get_normal_bcs(fnc_space, bc=None):
    fault = dfn.DirichletBC(fnc_space,
                        Constant(0.0),
                        fault_boundary)
    plate = dfn.DirichletBC(fnc_space,
                        Constant(params['plate_rate']),
                        plate_boundary)
    mantle = dfn.DirichletBC(fnc_space,
                         Constant(0.0),
                         mantle_boundary)
    return [fault, plate, mantle]

test_bc = TestBC()
test_bc.set_params(params['fault_depth'],
                 params['recur_interval'],
                 params['material']['shear_modulus'],
                 params['viscosity'],
                 params['plate_rate'])
test_bc.t = params['delta_t']
def get_test_bcs(fnc_space, bc=None):
    testing = dfn.DirichletBC(fnc_space,
                          bc,
                          testing_boundary)
    return [testing]
