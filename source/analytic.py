import numpy as np
from math import factorial
from scipy.special import gammainc


def elastic_stress(x, y, s, D, shear_modulus):
    """
    Use the elastic half-space stress solution from Segall (2010)
    """
    factor = (s * shear_modulus) / (2 * np.pi)
    main_term = -(y - D) / ((y - D) ** 2 + x ** 2)
    image_term = (y + D) / ((y + D) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)

    main_term = x / (x ** 2 + (y - D) ** 2)
    image_term = -x / (x ** 2 + (y + D) ** 2)
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
    term2 = -np.arctan((2 * m - 1 + y) / x)
    return factor * (term1 + term2)

def cm(x, y, tau, tau0, m, past_events):
    term1 = 0.0
    for k in range(0, past_events + 1):
        term1 += np.exp(-(tau + k * tau0)) * \
            (tau + k * tau0) ** (m - 1)
    term1 *= tau0 / factorial(m - 1)
    term2 = 1.0 - gammainc(m, tau + (past_events + 1) * tau0)
    term2 /= factorial(m - 1)
    return term1 + term2


def velocity(x, y, t, D, shear_modulus, viscosity, plate_rate):
    scaled_x = x / D
    scaled_y = y / D
    T = 10.0 * 3600 * 24 * 365
    images = 50
    past_events = 50
    tau = (shear_modulus * t) / (2 * viscosity)
    tau0 = (shear_modulus * T) / (2 * viscosity)
    vl = np.zeros_like(x)
    term1 = 0.0
    for m in range(1, images + 1):
        term = cm(scaled_x, scaled_y, tau, tau0, m, past_events) - 1.0
        term *= np.where(scaled_y > 1,
                         S_m_halfspace(scaled_x, scaled_y, m),
                         S_m_layer(x, scaled_y, m))
        term1 += term
    term2 = np.where(scaled_y > 1,
                     sum_halfspace(scaled_x, scaled_y),
                     sum_layer(scaled_x, scaled_y))
    vl = term1 + term2
    return np.double(vl * plate_rate)

if __name__ == "__main__":
    from matplotlib import pyplot as pyp
    X, Y = np.meshgrid(np.linspace(1, 2e4, 100),
                np.linspace(0, 2e4, 100))
    pyp.imshow(Y)
    pyp.colorbar()
    pyp.show()
    t = 0.0 * 3600 * 24 * 365
    v = velocity(X, Y, t, 1.0e4, 30.0e9, 1.0e19, 1.2e-9)
    # import pdb; pdb.set_trace()
    pyp.imshow(v, interpolation='none')
    pyp.colorbar()
    pyp.figure(2)
    pyp.contour(v)
    pyp.show()
