import numpy as np


def ws_transformation(Z, phi, w):
    return (np.exp(1j*phi)*w + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*w)


def ws_transformation_real_params(X, Y, phi, psi):
    return (np.exp(1j*(phi + psi)) + X + 1j*Y)/(1 + np.exp(1j*(phi + psi))*(X - 1j*Y))


def inverse_ws_transform(Z, phi, z):
    return np.exp(-1j*phi)*(z - Z)/(1 - np.conjugate(Z)*z)


def Z_dot_g(Z, omegas_Z, p_1, p_m1):
    return 1j*omegas_Z * Z + p_1 - p_m1 * Z**2


def z_dot_g(z, omegas_z, adj_matrix, z_and_zeta):
    n = len(z)
    q = (adj_matrix @ z_and_zeta)[0:n]
    q_tilde = (adj_matrix @ np.conj(z_and_zeta))[0:n]
    return 1j * omegas_z * z + q - z**2 * q_tilde


def phi_dot_g(Z, omegas_Z, p_m1):
    return omegas_Z - 2 * np.imag(p_m1 * Z)


def ws_equations_graph(t, state, w_allparts, omegas_Z, omegas_z, adj_matrix):
    # extract the variables from the arguments
    Z, phi, z = state

    # compute zeta, p_1, p_m1
    zeta = []
    for mu, Z_mu in enumerate(Z):
        zeta.append(ws_transformation(Z_mu, phi[mu], w_allparts[mu]))
    zeta = np.array([zeta]).T
    z_and_zeta = np.concatenate((z, zeta), axis=0)
    p_1 = adj_matrix @ z_and_zeta
    p_m1 = adj_matrix @ z_and_zeta**(-1)

    return [Z_dot_g(Z, omegas_Z, p_1, p_m1), phi_dot_g(Z, omegas_Z, p_m1), z_dot_g(z, omegas_z, adj_matrix, z_and_zeta)] # return list with Z_dot, phi_dot and z_dot







    # Z, phi, z = state
    # zeta = []
    # for mu, Z_mu in enumerate(Z):
        # zeta.append(ws_transformation(Z_mu, phi[mu], w_allparts[mu]))
    # zeta_and_z = np.concatenate(np.concatenate(zeta), z)

    # p_1 = np.sum(adj_matrix @ zeta_and_z)    # WARNING: Make sure that adj matrix is correctly formatted
    # p_m1 = np.sum(adj_matrix @ zeta_and_z**(-1))

    # return np.array([Z_dot(Z, omegas, p_1, p_m1), phi_dot(Z, omegas, p_m1)]) # TODO: ADD z_dot
