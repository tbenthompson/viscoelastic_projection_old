import numpy as np
from math import factorial
from scipy.special import gammainc, gamma


def elastic_stress(x, y, s, D, shear_modulus):
    """
    Initial stresses for the Savage (2000) Viscoelastic-Coupling
    Model.
    I should figure out whether this represents a left-lateral
    or right-lateral slip.
    """
    factor = (s * shear_modulus) / (2 * np.pi)
    main_term = (y - D) / ((y - D) ** 2 + x ** 2)
    image_term = -(y + D) / ((y + D) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)

    main_term = -x / (x ** 2 + (y - D) ** 2)
    image_term = x / (x ** 2 + (y + D) ** 2)
    Szy = factor * (main_term + image_term)
    return Szx, Szy



def sum_layer(x, y):
    factor = 1.0 / (2.0 * np.pi)
    term1 = -np.arctan((1 + y) / x)
    term2 = -np.arctan((1 - y) / x)
    term3 = np.pi * np.sign(x)
    return factor * (term1 + term2 + term3)

def sum_halfspace(x, y):
    factor = 1.0 / (2.0 * np.pi)
    term1 = -np.arctan((1 + y) / x)
    term2 = -np.arctan((-1 + y) / x)
    term3 = np.pi * np.sign(x)
    return factor * (term1 + term2 + term3)

def S_m_layer(x, y, m):
    factor = 1.0 / (2.0 * np.pi)
    term1 = np.arctan((2 * m + 1 + y) / x)
    term2 = -np.arctan((2 * m - 1 + y) / x)
    term3 = np.arctan((2 * m + 1 - y) / x)
    term4 = -np.arctan((2 * m - 1 - y) / x)
    return factor * (term1 + term2 + term3 + term4)

def S_m_halfspace(x, y, m):
    factor = 1.0 / (2.0 * np.pi)
    term1 = np.arctan((2 * m + 1 + y) / x)
    # There was a 3 here. Is that correct?
    term2 = -np.arctan((2 * m - 3 + y) / x)
    # term2 = -np.arctan((2 * m - 1 + y) / x)
    return factor * (term1 + term2)

def mathematica_gamma(a, z):
    return (1.0 - gammainc(a, z)) * gamma(a)

def cm(x, y, tau, tau0, m, past_events):
    term1 = 0.0
    for k in range(0, past_events + 1):
        term1 += np.exp(-(tau + k * tau0)) * \
            (tau + k * tau0) ** (m - 1)
    term1 *= tau0 / factorial(m - 1)
    term2 = mathematica_gamma(m, tau + (past_events + 1) * tau0)
    term2 /= factorial(m - 1)
    return term1 + term2


def velocity(x, y, tau, tau0):
    """
    Nondimensional version of "velocity_dimensional"

    tau = (mu * t) / (2 * eta)
    tau0 = (mu * T) / (2 * eta)
    """
    images = 50
    past_events = 50
    vl = np.zeros_like(x)
    term1 = 0.0
    for m in range(1, images + 1):
        term = cm(x, y, tau, tau0, m, past_events) - 1.0
        term *= np.where(y > 1,
                         S_m_halfspace(x, y, m),
                         S_m_layer(x, y, m))
        term1 += term
    term2 = np.where(y > 1,
                     sum_halfspace(x, y),
                     sum_layer(x, y))
    vl = term1 + term2
    return vl

def velocity_dimensional(X, Y, D, t, T, shear_modulus, viscosity, plate_rate):
    """
    Parameters are
    X, Y -- position with X as horizontal and Y positive downwards, in m
    D -- depth of fault and elastic layer, in m
    t -- time after last event, in secs
    T -- interevent time, in secs
    shear_modulus -- in Pa
    viscosity -- in Pa/sec
    plate_rate -- in m/sec

    Computes the velocity resulting from a screw dislocation offset.
    Python version of the Savage (2000) Mathematica script.
    The solution is derived by solving the corresponding elastic problem
    and using the Laplace domain correspondence principle. Linear
    superposition is used extensively.
    """
    tau = (shear_modulus * t) / (2 * viscosity)
    tau0 = (shear_modulus * T) / (2 * viscosity)
    x = X / D
    y = Y / D
    v = velocity(x, y, tau, tau0)
    return v * plate_rate

if __name__ == "__main__":
    from matplotlib import pyplot as pyp
    X, Y = np.meshgrid(np.linspace(1, 2e4, 300),
                np.linspace(0, 2e4, 300))
    t = 0.0 * 3600 * 24 * 365
    T = 0.0 * 3600 * 24 * 365
    shear_modulus = 3.0e10
    viscosity = 1.0e19
    D = 1.0e4
    v_plate = 1.0e-9

    v = velocity_dimensional(X, Y, D, t, T, shear_modulus, viscosity, v_plate)
    import pdb; pdb.set_trace()
    pyp.imshow(v, interpolation='none')
    pyp.colorbar()
    pyp.figure(2)
    pyp.contour(v)
    pyp.show()
