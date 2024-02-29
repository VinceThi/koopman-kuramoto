import numpy as np


def z_j(Z, phi, w_j):
    return (np.exp(1j * phi) * w_j + Z) / (1 + np.exp(1j * phi) * np.conjugate(Z) * w_j)

def w_j(Z, phi, z_j):
    return np.exp(- 1j * phi) * (z_j - Z) / (1 - np.conjugate(Z) * z_j)

def compute_FGF_bar(Z, phi, w, nat_freq, coupl_const, N):
    F_bar = (1j * coupl_const) / (2 * N) * np.sum([z_j(Z, phi, w_j) for w_j in w])
    G = nat_freq
    F = np.conjugate(F_bar)
    return F, G, F_bar

def Z_dot(Z, F, G, F_bar):
    return 1j * (F * Z ** 2 + G * Z + F_bar)

def phi_dot(Z, F, G, F_bar):
    return G + F * Z + F_bar * np.conjugate(Z)

def ws_equations(state, w, nat_freq, coupl_const, N):
    Z, phi = state
    F, G, F_bar = compute_FGF_bar(Z, phi, w, nat_freq, coupl_const, N)
    return np.array([Z_dot(Z, F, G, F_bar), phi_dot(Z, F, G, F_bar)])

