import numpy as np


def ws_transformation(Z, phi, w):
    """Apply Watanabe-Strogatz transformation to w with parameters Z and phi."""
    return (np.exp(1j*phi)*w + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*w)


def ws_transformation_real_params(X, Y, phi, psi):
    """Apply Watanabe-Strogatz transformation with real quantites."""
    return (np.exp(1j*(phi + psi)) + X + 1j*Y)/(1 + np.exp(1j*(phi + psi))*(X - 1j*Y))


def inverse_ws_transform(Z, phi, z):
    """Apply inverse Watanabe-Strogatz transformation to z with parameters Z and phi."""
    return np.exp(-1j*phi)*(z - Z)/(1 - np.conjugate(Z)*z)


def Z_dot_g(Z, omegas_Z, p_1, p_m1):
    """Dynamical equation for the Z variables in Watanabe-Strogatz reduction on a graph."""
    return 1j*omegas_Z * Z + p_1 - p_m1 * Z**2


def phi_dot_g(Z, omegas_Z, p_m1):
    """Dynamical equation for the phi variables in Watanabe-Strogatz reduction on a graph."""
    return omegas_Z - 2 * np.imag(p_m1 * Z)


def z_dot_g(z, coupling, omegas_z, adj_matrix, z_and_zeta):
    """Dynamical equations for the oscillators in the NIP for Watanabe-Strogatz reduction on a graph."""
    n = len(z)
    q = coupling/2 * (adj_matrix @ z_and_zeta)[:n]
    q_tilde = coupling/2 * (adj_matrix @ np.conj(z_and_zeta))[:n]
    return 1j * omegas_z * z + q - z**2 * q_tilde


def ws_equations_graph(t, state, w_allparts, coupling, omegas_Z, omegas_z, adj_matrix, adj_matrix_intparts):
    """Dynamical equation for the Watanabe-Strogatz reduced dynamics.

    Args:
        t (float): Time value (necessary for integration function).
        state (ndarray): Array containing the WS variables.
                         First elements are Z variables, then phi variables, then z variables.
        w (ndarray): Constants of the Watanabe-Strogatz transform. Each row corresponds to
                     a different PIP.
        coupling (float): Coupling constant.
        omegas_Z (list): Natural frequencies of the oscillators in each PIP.
        omegas_z (list): Natural frequencies of the oscillators in the NIP.
        adj_matrix (ndarray): Adjacency matrix of the graph.
        adj_matrix_intparts (ndarray): Adjacency matrix of the PIPs (one row per PIP).

    Returns:
        (ndarray): Array containing the derivatives of the reduced variables.
                   First elements are Z derivatives, then phi derivatives, then z derivatives.

    """

    # extract the variables from the 'state' argument
    Z = np.array([state[:len(omegas_Z)]]).T
    phi = np.array([state[len(omegas_Z):2*len(omegas_Z)]]).T
    z = np.array([state[-len(omegas_z):]]).T if len(omegas_z) != 0 else np.array([[]]).T

    # compute zeta, p_1, p_m1
    zeta = np.concatenate([ws_transformation(Z_mu, phi[mu], w_allparts[mu]).reshape(-1, 1) for mu, Z_mu in enumerate(Z)], axis=0)
    z_and_zeta = np.concatenate((z, zeta), axis=0)
    p_1 = coupling/2 * (adj_matrix_intparts @ z_and_zeta)
    p_m1 = coupling/2 * (adj_matrix_intparts @ z_and_zeta**(-1))

    # compute the derivatives
    derivatives = [Z_dot_g(Z, omegas_Z, p_1, p_m1), phi_dot_g(Z, omegas_Z, p_m1), z_dot_g(z, coupling, omegas_z, adj_matrix, z_and_zeta)]
    derivatives = np.concatenate(list(map(lambda x: x.flatten(), derivatives)))
    return derivatives


