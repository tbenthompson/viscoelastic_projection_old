import dolfin as dfn
from params import params

class TestBC(dfn.Expression):
    def set_params(self, t = 0.0):
        self.t = t

    def eval(self, value, x):
        value[0] = params['velocity'](x[0], x[1], self.t)

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
                        dfn.Constant(0.0),
                        fault_boundary)
    plate = dfn.DirichletBC(fnc_space,
                        dfn.Constant(params['plate_rate']),
                        plate_boundary)
    mantle = dfn.DirichletBC(fnc_space,
                         dfn.Constant(0.0),
                         mantle_boundary)
    return [fault, plate, mantle]

test_bc = TestBC()
test_bc.set_params(params['delta_t'])
def get_test_bcs(fnc_space, bc=None):
    testing = dfn.DirichletBC(fnc_space,
                          bc,
                          testing_boundary)
    return [testing]
