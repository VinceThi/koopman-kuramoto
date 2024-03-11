import numpy as np
from dynamics.watanabe_strogatz import ws_transformation
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
import pytest


def test_get_watanabe_strogatz_initial_conditions_and_w():
    N = 10
    np.random.seed(44)
    theta0 = 2*np.pi*np.random.random(N)
    z0 = np.exp(1j*theta0)
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N)
    assert np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-8)


# from dynamics.constants_of_motion import get_independent_cross_ratios_complete_graph
# from dynamics.constants_w import get_w
# def test_coherence_initial_conditions():
#     N = 100
#     theta0 = 2 * np.pi * np.random.random(N)
#     z0 = np.exp(1j * theta0)
#     cross_ratios = get_independent_cross_ratios_complete_graph(z0)
#     w = get_w(cross_ratios, N, nb_iter=500)
#     Z0, phi0 = get_Z0_phi0(theta0, w)
#     assert np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-10)


if __name__ == "__main__":
    pytest.main()
