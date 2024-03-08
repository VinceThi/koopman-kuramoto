import numpy as np


def ws_transformation(Z, phi, w):
    return (np.exp(1j*phi)*w + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*w)


def ws_transformation_real_params(X, Y, phi, psi):
    return (np.exp(1j*(phi + psi)) + X + 1j*Y)/(1 + np.exp(1j*(phi + psi))*(X - 1j*Y))


def inverse_ws_transform(Z, phi, z):
    return np.exp(-1j*phi)*(z - Z)/(1 - np.conjugate(Z)*z)


def Z_dot(Z, F, G, F_bar):
    return 1j*(F*Z**2 + G*Z + F_bar)


def phi_dot(Z, F, G, F_bar):
    return G + F*Z + F_bar*np.conjugate(Z)


def ws_equations_kuramoto(t, state, w, omega, coupling, N):
    Z, phi = state
    F_bar = coupling/(2*1j*N)*np.sum(ws_transformation(Z, phi, w))
    G = omega
    F = np.conjugate(F_bar)
    return np.array([Z_dot(Z, F, G, F_bar), phi_dot(Z, F, G, F_bar)])
