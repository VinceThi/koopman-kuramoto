import numpy as np
from scipy.optimize import root
from tqdm import tqdm
from dynamics.ws_initial_conditions import *


def get_ws_initial_conditions_graph(theta0_allparts, N, dispersed_guess=False, nb_guess=5000, tol=1e-10):
    Z = []
    phi = []
    w = []
    for theta0 in theta0_allparts:
        Z_mu, phi_mu, w_mu = get_watanabe_strogatz_initial_conditions(theta0, N, dispersed_guess=dispersed_guess, nb_guess=nb_guess, tol=tol)
        Z.append(Z_mu)
        phi.append(phi_mu)
        w.append(w_mu)
    Z = np.array([Z]).T
    phi = np.array([phi]).T
    w = np.array(w)

    return Z, phi, w




