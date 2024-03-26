import numpy as np
from dynamics.ws_initial_conditions import *


def get_ws_initial_conditions_graph(theta0_allparts, non_integrable_part=True, dispersed_guess=False, nb_guess=5000, tol=1e-10):
    Z = []
    phi = []
    w = []
    theta0_intparts = theta0_allparts
    if non_integrable_part:
        theta0_intparts = theta0_intparts[1:]
    for i, theta0 in enumerate(theta0_intparts[:]):
        print(f"computing initial conditions for part {i+1}")
        Z_mu, phi_mu, w_mu = get_watanabe_strogatz_initial_conditions(theta0, len(theta0), dispersed_guess=dispersed_guess, nb_guess=nb_guess, tol=tol)
        Z.append(Z_mu)
        phi.append(phi_mu)
        w.append(w_mu)
    Z = np.array(Z)
    phi = np.array(phi)

    return Z, phi, w




