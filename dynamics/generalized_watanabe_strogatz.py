import numpy as np


def ws_transformation(Z, phi, w):
    return (np.exp(1j*phi)*w + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*w)


def ws_transformation_real_params(X, Y, phi, psi):
    return (np.exp(1j*(phi + psi)) + X + 1j*Y)/(1 + np.exp(1j*(phi + psi))*(X - 1j*Y))


def inverse_ws_transform(Z, phi, z):
    return np.exp(-1j*phi)*(z - Z)/(1 - np.conjugate(Z)*z)


def Z_dot(Z, omegas, p_1, p_m1):
    return 1j*omegas * Z + p_1 - p_m1 * Z**2


def z_dot(z, omegas, adj_submatrix, zeta_and_z):
    # TODO
    pass


def phi_dot(Z, omegas, p_m1):
    return omegas - 2 * np.imag(p_m1 * Z)


def ws_equations_kuramoto(t, state, w_allparts, omegas, adj_matrix, N):
    Z, phi, z = state
    zeta = []
    for mu, Z_mu in enumerate(Z):
        zeta.append(ws_transformation(Z_mu, phi[mu], w_allparts[mu]))
    zeta_and_z = np.concatenate(np.concatenate(zeta), z)

    p_1 = np.sum(adj_matrix @ zeta_and_z)    # WARNING: Make sure that adj matrix is correctly formatted
    p_m1 = np.sum(adj_matrix @ zeta_and_z**(-1))

    return np.array([Z_dot(Z, omegas, p_1, p_m1), phi_dot(Z, omegas, p_m1)]) # TODO: ADD z_dot

