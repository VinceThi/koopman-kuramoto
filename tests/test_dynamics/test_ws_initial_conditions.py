import numpy as np
from dynamics.watanabe_strogatz import ws_transformation
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from tqdm import tqdm


def test_get_watanabe_strogatz_initial_conditions():
    N = 100
    nb_initial_conditions = 1000
    success_counter = 0
    for _ in tqdm(range(nb_initial_conditions)):
        theta0 = 2*np.pi*np.random.random(N)
        z0 = np.exp(1j*theta0)
        Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0)
        # print("|z0 - ws_transformation(Z0, phi0, w)| = ", np.abs(z0 - ws_transformation(Z0, phi0, w)))
        success_counter += int(np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-6))
        """ 
        Since the objective function is squared, the tolerance 1e-10 specified in 
        get_watanabe_strogatz_initial_conditions is smaller than the actual error 
        np.abs(z0 - ws_transformation(Z0, phi0, w)). This is why we choose a tolerance of 1e-6.
        """
    if success_counter == nb_initial_conditions:
        print("The test was successful !")
    else:
        print("The test failed.")


test_get_watanabe_strogatz_initial_conditions()
