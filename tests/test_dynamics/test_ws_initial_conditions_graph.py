import numpy as np
from dynamics.watanabe_strogatz import ws_transformation
from dynamics.ws_initial_conditions_graph import get_ws_initial_conditions_graph
import pytest


def test_get_ws_initial_conditions_graph():
    print("\nBeginning test_get_ws_initial_conditions_graph ...")
    N = 50
    np.random.seed(42)
    theta0 = [2*np.pi*np.random.random(int(N/2) - 1), 2*np.pi*np.random.random(N - int(N/2) + 1)]
    z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0]
    Z0, phi0, w = get_ws_initial_conditions_graph(theta0, non_integrable_part=False, nb_guess=10000)
    err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0)]
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)
    assert np.all([np.all(e < 1e-8) for e in err])


def test_get_ws_initial_conditions_graph_nonintpart():
    print("\nBeginning test_get_ws_initial_conditions_graph_nonintpart ...")
    N = 50
    np.random.seed(42)
    theta0 = [2*np.pi*np.random.random(3), 2*np.pi*np.random.random(int(N/2) - 4), 2*np.pi*np.random.random(N - int(N/2) + 1)]
    z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0[1:]]
    Z0, phi0, w = get_ws_initial_conditions_graph(theta0, nb_guess=10000)
    err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0)]
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)
    assert np.all([np.all(e < 1e-8) for e in err])


if __name__ == "__main__":
    pytest.main()
