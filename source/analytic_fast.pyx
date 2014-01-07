# encoding: utf-8
# cython: profile=True
import numpy as np
import scipy.integrate
from libc.math cimport exp, pow, atan, cos
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

def simple_velocity(double x, double y, double t):
    from params import params
    cdef int images = 20
    cdef double D = params['fault_depth']
    cdef double mu = params['material']['shear_modulus'] 
    cdef double eta = params['viscosity']
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

def cosine_slip_fnc(double D):
    def slip_fnc(double z):
        if z > D:
            return 0.0
        return pow(cos(z * np.pi / 20000.0), 2)
    return slip_fnc

def integral_stress(double x, double y, s):
    from params import params
    cdef double D = params['fault_depth']
    cdef double shear_modulus = params['material']['shear_modulus']
    cdef double factor, main_term, image_term, Szx, Szy
    factor = (shear_modulus) / (2 * np.pi)
    def main_term_fnc(double z):
        return s(z) * ((y - z) ** 2 - x ** 2) / \
            (((y - z) ** 2 + x ** 2) ** 2)
    def image_term_fnc(double z):
        return s(z) * ((y + z) ** 2 - x ** 2) / \
            (((y + z) ** 2 + x ** 2) ** 2)
    main_term = scipy.integrate.quad(main_term_fnc, 0, D)[0]
    image_term = scipy.integrate.quad(image_term_fnc, 0, D)[0]
    Szx = factor * (main_term + image_term)

    def main_term_fnc(double z):
        return s(z) * (-2 * x * (y - z)) / \
            (((y - z) ** 2 + x ** 2) ** 2)
    def image_term_fnc(double z):
        return s(z) * (-2 * x * (y + z)) / \
            (((y + z) ** 2 + x ** 2) ** 2)
    main_term = scipy.integrate.quad(main_term_fnc, 0, D)[0]
    image_term = scipy.integrate.quad(image_term_fnc, 0, D)[0]
    Szy = factor * (main_term + image_term)
    return Szx, Szy

def integral_velocity(double x, double y, double t, s):
    from params import params
    cdef int images = 20
    cdef double D = params['fault_depth']
    cdef double mu = params['material']['shear_modulus'] 
    cdef double eta = params['viscosity']
    cdef double v = 0.0
    cdef double t_r = (2 * eta) / mu
    cdef int m = 0
    cdef double factor
    cdef double term, term1, term2, term3, term4
    for m in range(1, images):
        factor = (1.0 / (2.0 * PI)) * pow(t / t_r, m - 1) / factorial(m - 1)
        if y > D:
            def term_1_fnc(double z):
                return s(z) * (2 * m + 1) / \
                    (x * (1 + (((2 * m + 1) * z + y) / x) ** 2))
            def term_2_fnc(double z):
                return -s(z) * (2 * m - 3) / \
                    (x * (1 + (((2 * m - 3) * z + y) / x) ** 2))
            term1 = scipy.integrate.quad(term_1_fnc, 0, D)[0]
            term2 = scipy.integrate.quad(term_2_fnc, 0, D)[0]
            term = term1 + term2
        else:
            def term_1_fnc(double z):
                return s(z) * (2 * m + 1) / \
                    (x * (1 + (((2 * m + 1) * z + y) / x) ** 2))
            def term_2_fnc(double z):
                return -s(z) * (2 * m - 1) / \
                    (x * (1 + (((2 * m - 1) * z + y) / x) ** 2))
            def term_3_fnc(double z):
                return s(z) * (2 * m + 1) / \
                    (x * (1 + (((2 * m + 1) * z - y) / x) ** 2))
            def term_4_fnc(double z):
                return -s(z) * (2 * m - 1) / \
                    (x * (1 + (((2 * m - 1) * z - y) / x) ** 2))
            term1 = scipy.integrate.quad(term_1_fnc, 0, D)[0]
            term2 = scipy.integrate.quad(term_2_fnc, 0, D)[0]
            term3 = scipy.integrate.quad(term_3_fnc, 0, D)[0]
            term4 = scipy.integrate.quad(term_4_fnc, 0, D)[0]
            #term1 = atan(((2 * m + 1) * D + y) / x)
            #term2 = -atan(((2 * m - 1) * D + y) / x)
            #term3 = atan(((2 * m + 1) * D - y) / x)
            #term4 = -atan(((2 * m - 1) * D - y) / x)

            term = term1 + term2 + term3 + term4
        v += factor * term
    v *= exp(-t / t_r) * (1.0 / t_r)
    return v
