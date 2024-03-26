import numpy as np
import time
from tests.optimization.ws_initconds_dual_annealing import *
from dynamics.watanabe_strogatz import ws_transformation


print("\nBeginning 'dual annealing' method...")
N = 10
np.random.seed(42)
beginning = time.time()
for _ in range(10):
    theta0 = 2*np.pi*np.random.random(N)
    z0 = np.exp(1j*theta0)
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, nb_guess=10000)
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", np.abs(z0 - ws_transformation(Z0, phi0, w)))
    assert np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-8)
end = time.time()
print(f"Computation completed in {end - beginning}")
