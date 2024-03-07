import numpy as np
from scipy.optimize import root


def ws_transformation_real_params(X, Y, phi, w):
    return (np.exp(1j*phi)*w + X + 1j*Y)/(1 + np.exp(1j*phi)*(X - 1j*Y)*w)


def objective_function_Z_phi(x, init_z, w):
    """ X, Y, phi = x """
    complex_valued_function = ws_transformation_real_params(x[0], x[1], x[2], w[:3]) - init_z[:3]
    real_part = np.real(complex_valued_function)
    imag_part = np.imag(complex_valued_function)
    return np.concatenate([real_part, imag_part])


def get_Z0_phi0(init_z, w, nb_iter=500):
    for i in range(nb_iter):
        X = np.random.uniform(-1, 1)
        Y = np.random.uniform(-1, 1)
        phi = 2*np.pi*np.random.random()
        print(X, Y, phi, objective_function_Z_phi(np.array([X, Y, phi]), init_z, w))
        solution = root(objective_function_Z_phi, np.array([X, Y, phi]), args=(init_z, w), method='hybr', tol=1e-10)
        if solution.success:
            break
    if not solution.success:
        raise ValueError("The optimization did not converge to successful values of Z(0) and phi(0).")
    X0, Y0, phi0 = solution.x
    return X0 + 1j*Y0, phi0

from constants_w import *
from constants_of_motion import *
N = 100
theta0 = 2 * np.pi * np.random.random(N)
z0 = np.exp(1j * theta0)
cross_ratios = get_independent_cross_ratios_complete_graph(z0)
w = get_w(cross_ratios, N, nb_iter=500)
Z0, phi0 = get_Z0_phi0(z0, w)
print(Z0, phi0)