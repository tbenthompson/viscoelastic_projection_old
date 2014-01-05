# encoding: utf-8
# cython: profile=True
# filename: calc_pi.pyx
import numpy as np
from libc.math cimport exp, pow, atan
from math import factorial

cdef double PI = 3.14159265358979

def simple_stress(double x, double y):
    from params import params
    cdef double s = params['fault_slip']
    cdef double D = params['fault_depth']
    cdef double shear_modulus = params['material']['shear_modulus']
    cdef double factor, main_term, image_term, Szx, Szy
    factor = (s * shear_modulus) / (2 * np.pi)
    main_term = (y - D) / ((y - D) ** 2 + x ** 2)
    image_term = -(y + D) / ((y + D) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)

    main_term = -x / (x ** 2 + (y - D) ** 2)
    image_term = x / (x ** 2 + (y + D) ** 2)
    Szy = factor * (main_term + image_term)
    return Szx, Szy

def simple_velocity(double x, double y, double D, double t, double mu,
                    double eta, double v_plate, int images=20):
    cdef double v = 0.0
    cdef double x_scaled = x / D
    cdef double y_scaled = y / D
    cdef double t_r = (2 * eta) / mu
    cdef int m = 0
    cdef double factor
    cdef double term, term1, term2, term3, term4
    for m in range(1, images):
        factor = pow(t / t_r, m - 1) / factorial(m - 1)
        if y_scaled > 1:
            term1 = atan((2 * m + 1 + y_scaled) / x_scaled)
            term2 = -atan((2 * m - 3 + y_scaled) / x_scaled)
            term = (1.0 / (2.0 * PI)) * (term1 + term2)
        else:
            term1 = atan((2 * m + 1 + y_scaled) / x_scaled)
            term2 = -atan((2 * m - 1 + y_scaled) / x_scaled)
            term3 = atan((2 * m + 1 - y_scaled) / x_scaled)
            term4 = -atan((2 * m - 1 - y_scaled) / x_scaled)
            term = (1.0 / (2.0 * PI)) * (term1 + term2 + term3 + term4)
        v += factor * term
    v *= exp(-t / t_r) * (1.0 / t_r)
    return v
