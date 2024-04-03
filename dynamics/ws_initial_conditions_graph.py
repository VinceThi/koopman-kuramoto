import numpy as np
from dynamics.ws_initial_conditions import *


def get_ws_initial_conditions_graph(theta0_allparts, dispersed_guess=False, nb_guess=1, tol=1e-10):
    """ Apply get_watanabe_strogatz_initial_conditions to each PIP.

    Args:
        theta0_allparts (ndarray): Initial conditions of the oscillators.
                            Each row corresponds to a different PIP.
        dispersed_guess (bool): If True, initial guesses for R are restrained to 0 < R < 0.2
                                instead of 0 < R < 1. Defaults to False.
        nb_guess (int): Maximum number of sets of initial guesses for the
                        optimization process. Defaults to 1.
        tol (float): parameter 'xtol' for 'hybr' method of scipy.optimize.root.
                     Defaults to 1e-10.

    Returns:
        (tuple of ndarrays): Initial conditions for Z variables, initial conditions for phi
                             and w constants.

    """
    Z = []
    phi = []
    w = []
    theta0_intparts = theta0_allparts[1:]
    for i, theta0 in enumerate(theta0_intparts[:]):
        print(f"computing initial conditions for part {i+1}")
        Z_mu, phi_mu, w_mu = get_watanabe_strogatz_initial_conditions(theta0, dispersed_guess=dispersed_guess, nb_guess=nb_guess, tol=tol)
        Z.append(Z_mu)
        phi.append(phi_mu)
        w.append(w_mu)
    Z = np.array(Z)
    phi = np.array(phi)

    return Z, phi, w




