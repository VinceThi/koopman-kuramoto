import numpy as np


def ws_transformation(Z, phi, w):
    """Apply Watanabe-Strogatz transformation to w with parameters Z and phi."""
    return (np.exp(1j*phi)*w + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*w)


def inverse_ws_transform(Z, phi, z):
    """Apply inverse Watanabe-Strogatz transformation to z with parameters Z and phi."""
    return np.exp(-1j*phi)*(z - Z)/(1 - np.conjugate(Z)*z)


def Z_dot(Z, F, G, F_bar):
    """Dynamical equation for the Z variable in Watanabe-Strogatz reduction."""
    return 1j*(F*Z**2 + G*Z + F_bar)


def phi_dot(Z, F, G, F_bar):
    """Dynamical equation for the phi variable in Watanabe-Strogatz reduction."""
    return G + F*Z + F_bar*np.conjugate(Z)


def ws_equations_kuramoto(t, state, w, coupling, omega):
    """Dynamical equation for the Watanabe-Strogatz reduced dynamics.

    Args:
        t (float): Time value (necessary for integration function).
        state (ndarray): Array containing the WS variables.
                         First element is Z, second element is phi.
        w (ndarray): Constants of the Watanabe-Strogatz transform.
        coupling (float): Coupling constant.
        omega (float): Natural frequency of the oscillators.

    Returns:
        (ndarray): Array containing the derivatives of the WS variables.
                   First element is Z_dot, second element is phi_dot.

    """
    Z, phi = state
    F_bar = coupling/(2*1j)*np.sum(ws_transformation(Z, phi, w))
    G = omega
    F = np.conjugate(F_bar)
    return np.array([Z_dot(Z, F, G, F_bar), phi_dot(Z, F, G, F_bar)])

