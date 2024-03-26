import numpy as np
import time
from tests.optimization.ws_initconds_ReIm import *
from dynamics.watanabe_strogatz import ws_transformation


print("\nBeginning test runs for ROOT function with 'hybr' algorithm. Objective function is ReIm.")
N = 300
# np.random.seed(42)
beginning = time.time()
for _ in range(1):
    theta0 = 2*np.pi*np.random.random(N)
    z0 = np.exp(1j*theta0)
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, nb_guess=1000, tol=1e-10)
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", (z0 - ws_transformation(Z0, phi0, w)))
    assert np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-6)
end = time.time()
print(f"Computation completed in {end - beginning}")


