import numpy as np
from math import factorial
from scipy.special import gammainc, gamma
from matplotlib import pyplot as pyp

##############################################################
# Stress solution
# This has big problems that need to be resolved. Maybe I
# should just interaface with Phoebe's code or something available online.
##############################################################
def steady_creep_dx(x, y):
    factor = 1.0 / (2 * np.pi)
    main_term = (y - 1) / ((y - 1) ** 2 + x ** 2)
    image_term = -(y + 1) / ((y + 1) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)
    return Szx


def steady_creep_dy(x, y):
    factor = 1.0 / (2 * np.pi)
    main_term = -x / (x ** 2 + (y - 1) ** 2)
    image_term = x / (x ** 2 + (y + 1) ** 2)
    Szy = factor * (main_term + image_term)
    return Szy


def S_m_layer_dx(x, y, m):
    factor = 1 / (2 * np.pi)
    term1 = -(2 * m + 1 + y) / ((2 * m + 1 + y) ** 2 + x ** 2)
    term2 = (2 * m - 1 + y) / ((2 * m - 1 + y) ** 2 + x ** 2)
    term3 = -(2 * m + 1 - y) / ((2 * m + 1 - y) ** 2 + x ** 2)
    term4 = (2 * m - 1 - y) / ((2 * m - 1 - y) ** 2 + x ** 2)
    return factor * (term1 + term2 + term3 + term4)


def S_m_layer_dy(x, y, m):
    factor = 1 / (2 * np.pi)
    term1 = x / ((2 * m + 1 + y) ** 2 + x ** 2)
    term2 = -x / ((2 * m - 1 + y) ** 2 + x ** 2)
    term3 = -x / ((2 * m + 1 - y) ** 2 + x ** 2)
    term4 = x / ((2 * m - 1 - y) ** 2 + x ** 2)
    return factor * (term1 + term2 + term3 + term4)


def S_m_halfspace_dx(x, y, m):
    factor = 1 / (2 * np.pi)
    term1 = -(2 * m + 1 + y) / ((2 * m + 1 + y) ** 2 + x ** 2)
    term2 = (2 * m - 3 + y) / ((2 * m - 3 + y) ** 2 + x ** 2)
    return factor * (term1 + term2)


def S_m_halfspace_dy(x, y, m):
    factor = 1 / (2 * np.pi)
    term1 = x / ((2 * m + 1 + y) ** 2 + x ** 2)
    term2 = -x / ((2 * m - 3 + y) ** 2 + x ** 2)
    return factor * (term1 + term2)


def A_m(tau, m):
    return gammainc(m, tau)


def stress(x, y, tau, tau0, past_events=50, images=50):
    """
    Initial stresses for the Savage (2000) Viscoelastic-Coupling
    Model.
    I should figure out whether this represents a left-lateral
    or right-lateral slip.
    """
    steady_term_dx = past_events * steady_creep_dx(x, y)
    steady_term_dy = past_events * steady_creep_dy(x, y)
    evolve_term_dx = np.zeros_like(x)
    evolve_term_dy = np.zeros_like(x)
    for m in range(1, images):
        Am = 0.0
        for k in range(past_events):
            Am += A_m(tau + k * tau0, m)
            if np.isnan(Am):
                raise Exception(k)
        Smdx = np.where(y > 1, S_m_halfspace_dx(x, y, m),
                        S_m_layer_dx(x, y, m))
        Smdy = np.where(y > 1, S_m_halfspace_dy(x, y, m),
                        S_m_layer_dy(x, y, m))
        evolve_term_dx += Am * Smdx
        evolve_term_dy += Am * Smdy
    Szx = steady_term_dx + evolve_term_dx
    Szy = steady_term_dy + evolve_term_dy
    return Szx, Szy

def stress_dimensional(x, y, D, t, T, shear_modulus, viscosity, v_plate, past_events=50, images=50):
    x = x / D
    y = y / D
    tau = (shear_modulus * t) / (2 * viscosity)
    tau0 = (shear_modulus * T) / (2 * viscosity)
    Szx, Szy = stress(x, y, tau, tau0, past_events=past_events, images=images)
    slip = v_plate * T
    factor = (shear_modulus * slip) / D
    return factor * Szx, factor * Szy

def test_stress():
    X, Y = np.meshgrid(np.linspace(1, 1e4, 50),
                       np.linspace(0, 2e4, 50))
    t = 0.0 * 3600 * 24 * 365
    T = 100.0 * 3600 * 24 * 365
    shear_modulus = 3.0e10
    viscosity = 5.0e19
    D = 1.0e4
    v_plate = 1.0e-9

    Szx, Szy = stress_dimensional(X, Y, D, t, T, shear_modulus, viscosity, v_plate)
    Szx2, Szy2 = simple_stress(X, Y, 1.0, D, shear_modulus)
    pyp.imshow(Szy, interpolation='none')
    pyp.colorbar()
    pyp.figure(2)
    pyp.imshow(np.log(np.abs(Szy)))
    pyp.colorbar()
    pyp.figure(3)
    pyp.imshow(Szy2, interpolation='none')
    pyp.colorbar()
    pyp.figure(4)
    pyp.imshow(np.log(np.abs(Szy2)))
    pyp.colorbar()
    pyp.show()




###############################################################
# Velocity solution
###############################################################
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


def velocity(x, y, tau, tau0, past_events=50, images=50):
    """
    Nondimensional version of "velocity_dimensional"

    tau = (mu * t) / (2 * eta)
    tau0 = (mu * T) / (2 * eta)
    """
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


def velocity_dimensional(X, Y, D, t, T, shear_modulus, viscosity, plate_rate, past_events=50, images=50):
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
    superposition is used extensively to represent a whole series of
    earthquakes rather than just one isolated event.
    """
    tau = (shear_modulus * t) / (2 * viscosity)
    tau0 = (shear_modulus * T) / (2 * viscosity)
    x = X / D
    y = Y / D
    v = velocity(x, y, tau, tau0, past_events=past_events, images=images)
    return v * plate_rate


def test_velocity():
    X, Y = np.meshgrid(np.linspace(1, 2e4, 300),
                       np.linspace(0, 2e4, 300))
    t = 0.1 * 3600 * 24 * 365
    T = 100.0 * 3600 * 24 * 365
    shear_modulus = 3.0e10
    viscosity = 5.0e19
    D = 1.0e4
    v_plate = 1.2e-9

    # v = velocity_dimensional(X, Y, D, t, T, shear_modulus, viscosity, v_plate)
    v = simple_velocity(X, Y, D, t, shear_modulus, viscosity, v_plate)
    Szx, Szy = simple_stress(X, Y, 1.0, D, shear_modulus)
    pyp.imshow(v, interpolation='none')
    pyp.colorbar()
    pyp.figure(5)
    pyp.plot(v[0, :])
    # pyp.figure(2)
    # pyp.imshow(Szx, interpolation='none')
    # pyp.colorbar()
    # pyp.figure(3)
    # pyp.imshow(Szy, interpolation='none')
    # pyp.colorbar()
    # divS = (Szx[1:, :] - Szx[:-1, :])[:,1:] + (Szy[:, 1:] - Szy[:, :-1])[1:, :]
    # divS /= (2e4 / 300.0)
    # print np.mean(divS)
    # pyp.figure(4)
    # pyp.imshow(divS, interpolation='none')
    # pyp.colorbar()
    pyp.show()


########################################################################
# Simple versions for one isolated earthquake.
########################################################################
def simple_stress(x, y, s, D, shear_modulus):
    factor = (s * shear_modulus) / (2 * np.pi)
    main_term = (y - D) / ((y - D) ** 2 + x ** 2)
    image_term = -(y + D) / ((y + D) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)

    main_term = -x / (x ** 2 + (y - D) ** 2)
    image_term = x / (x ** 2 + (y + D) ** 2)
    Szy = factor * (main_term + image_term)
    return Szx, Szy

def simple_velocity(x, y, D, t, mu, eta, v_plate, images=50):
    v = np.zeros_like(x)
    x_scaled = x / D
    y_scaled = y / D
    t_r = (2 * eta) / mu
    for m in range(1, images):
        factor = ((t / t_r) ** (m - 1)) / factorial(m - 1)
        term = np.where(y_scaled > 1,
                        S_m_halfspace(x_scaled, y_scaled, m),
                        S_m_layer(x_scaled, y_scaled, m))
        v += factor * term
    v *= np.exp(-t / t_r)
    v *= 1.0 / t_r
    return v

if __name__ == "__main__":
    test_velocity()
    # test_stress()
