import numpy as np


def ws_transformation(Z, phi, w):
    return (np.exp(1j*phi)*w + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*w)


def ws_transformation_real_params(X, Y, phi, psi):
    return (np.exp(1j*(phi + psi)) + X + 1j*Y)/(1 + np.exp(1j*(phi + psi))*(X - 1j*Y))


def inverse_ws_transform(Z, phi, z):
    return np.exp(-1j*phi)*(z - Z)/(1 - np.conjugate(Z)*z)


def Z_dot_g(Z, omegas_Z, p_1, p_m1):
    return 1j*omegas_Z * Z + p_1 - p_m1 * Z**2


def phi_dot_g(Z, omegas_Z, p_m1):
    return omegas_Z - 2 * np.imag(p_m1 * Z)


def z_dot_g(z, coupling, omegas_z, adj_matrix, z_and_zeta):
    n = len(z)
    q = coupling/2 * (adj_matrix @ z_and_zeta)[:n]
    q_tilde = coupling/2 * (adj_matrix @ np.conj(z_and_zeta))[:n]
    return 1j * omegas_z * z + q - z**2 * q_tilde


def ws_equations_graph(t, state, w_allparts, coupling, omegas_Z, omegas_z, adj_matrix, adj_matrix_intparts):
    # extract the variables from the arguments
    Z = np.array([state[:len(omegas_Z)]]).T
    phi = np.array([state[len(omegas_Z):2*len(omegas_Z)]]).T
    z = np.array([state[-len(omegas_z):]]).T if len(omegas_z) != 0 else np.array([[]]).T

    # compute zeta, p_1, p_m1
    zeta = []
    for mu, Z_mu in enumerate(Z):
        zeta_partmu = ws_transformation(Z_mu, phi[mu], w_allparts[mu])
        zeta += zeta_partmu.tolist()
    zeta = np.array([zeta]).T
    z_and_zeta = np.concatenate((z, zeta), axis=0)
    p_1 = coupling/2 * (adj_matrix_intparts @ z_and_zeta)
    p_m1 = coupling/2 * (adj_matrix_intparts @ z_and_zeta**(-1))

    derivatives = [Z_dot_g(Z, omegas_Z, p_1, p_m1), phi_dot_g(Z, omegas_Z, p_m1), z_dot_g(z, coupling, omegas_z, adj_matrix, z_and_zeta)]
    derivatives = np.concatenate(list(map(lambda x: x.flatten(), derivatives)))
    return derivatives


