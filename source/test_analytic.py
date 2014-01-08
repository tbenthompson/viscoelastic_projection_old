import numpy as np
import scipy.integrate
from analytic_fast import simple_stress, simple_velocity, \
                          integral_stress, integral_velocity, \
                          cosine_slip_fnc, constant_slip_fnc
import matplotlib.pyplot as pyp
from math import factorial, atan

def test_integral_stress():
    x = np.linspace(2.0, 10000, 100)
    y = np.linspace(0, 20000, 100)

    def slip_fnc(z):
        if z > 10000:
            return 0.0
        return 1.0

    # Szx = np.zeros((100, 100))
    # Szy = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            # Szx[i, j], Szy[i, j] = integral_stress(x[i], y[j], slip_fnc)
            Szx, Szy = integral_stress(x[i], y[j], slip_fnc)
            trueSzx, trueSzy = simple_stress(x[i], y[j])
            np.testing.assert_allclose(Szx, trueSzx, 1e-5)
            np.testing.assert_allclose(Szy, trueSzy, 1e-5)

def test_integral_velocity():
    x = np.linspace(2.0, 10000, 20)
    y = np.linspace(0, 20000, 20)
    print 1

    slip_fnc = constant_slip_fnc(10000.0)
    for i in range(20):
        for j in range(20):
            v = integral_velocity(x[i], y[j], 3.0e7, slip_fnc)
            true_v = simple_velocity(x[i], y[j], 3.0e7)
            print v, true_v
            np.testing.assert_allclose(v, true_v, 1e-5)


def test_visually():
    from params import params
    params['fault_depth'] = 10000.0
    nx = 100
    ny = 50
    x = np.linspace(5.0, 1e5, nx)
    y = np.linspace(0.0e4, 3e4, ny)
    # slip_fnc = constant_slip_fnc(10000.0)
    slip_fnc = cosine_slip_fnc(10000.0)
    # def slip_fnc(z):
    #     if z > 10000:
    #         return 0.0
    #     if z > 9000:
    #         return 0.5
    #     return 1.0

    Szx = np.zeros((nx, ny))
    Szy = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    true_v = np.zeros((nx, ny))
    # term1 = np.zeros((nx, ny))
    # term2 = np.zeros((nx, ny))
    # term3 = np.zeros((nx, ny))
    # term4 = np.zeros((nx, ny))
    fig = pyp.figure()
    ax = fig.add_subplot(111)
    obj = ax.imshow(v.T, interpolation='none')
    fig.show()
    for t in np.linspace(0, 1000 * 24 * 365 * 3600, 5):
        for i in range(nx):
            for j in range(ny):
                # v[i, j], term1[i, j], term2[i, j], term3[i, j], term4[i, j] = \
                v[i, j] = \
                    integral_velocity(x[i], y[j], t, slip_fnc, images=10)
                # true_v[i, j] = simple_velocity(x[i], y[j], 3.0e7)
            # Szx[i, j], Szy[i, j] = integral_stress(x[i], y[j], slip_fnc)
    # pyp.figure(1)
    # pyp.imshow(Szx.T, interpolation='none')
    # pyp.figure(2)
    # pyp.imshow(Szy.T, interpolation='none')
        # pyp.figure(3)
        # im = pyp.imshow(true_v.T, interpolation='none')
        # cb = pyp.colorbar()
        # ccc = cb.get_clim()
        pyp.imshow(v.T)
        pyp.draw()
        print "hello"
        # im2.set_clim(ccc)
    # pyp.figure(5)
    # pyp.imshow(term1.T, interpolation='none')
    # pyp.colorbar()
    # pyp.figure(6)
    # pyp.imshow(term2.T, interpolation='none')
    # pyp.colorbar()
    # pyp.figure(7)
    # pyp.imshow(term3.T, interpolation='none')
    # pyp.colorbar()
    # pyp.figure(8)
    # pyp.imshow(term4.T, interpolation='none')
    # pyp.colorbar()
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_visually()
