import numpy as np
from tests.optimization.ws_initial_conditions_optimization_tests import *
from dynamics.watanabe_strogatz import ws_transformation


print("\nBeginning test_get_watanabe_strogatz_initial_conditions...")
N = 10
np.random.seed(42)
theta0 = 2*np.pi*np.random.random(N)
z0 = np.exp(1j*theta0)
Z0, phi0, w = get_watanabe_strogatz_initial_conditions_ARCTAN(theta0, nb_guess=10000)
print("|z0 - ws_transformation(Z0, phi0, w)| = ", np.abs(z0 - ws_transformation(Z0, phi0, w)))
assert np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-8)
