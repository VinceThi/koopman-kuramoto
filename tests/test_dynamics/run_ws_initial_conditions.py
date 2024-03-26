import numpy as np
from dynamics.watanabe_strogatz import ws_transformation
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions


print("\nBeginning test_get_watanabe_strogatz_initial_conditions...")
N = 1000
np.random.seed(20)
theta0 = 2*np.pi*np.random.random(N)
z0 = np.exp(1j*theta0)
Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N, nb_guess=100)
print("|z0 - ws_transformation(Z0, phi0, w)| = ", np.abs(z0 - ws_transformation(Z0, phi0, w)))
assert np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-6)
