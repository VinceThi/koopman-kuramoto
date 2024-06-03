import numpy as np
from scipy.optimize import root


def objective_function_init_cond(x, theta0):
    """ Objective function for the optimization of WS initial conditions and constants.

    x (list): [R, Theta, Phi, psi_1, ..., psi_N],
              where Z = R e^(i Theta), w_j = e^(i psi_j) and Phi = Theta - phi.
    theta0 (ndarray): Initial conditions of the oscillators.

    """
    return np.concatenate([(np.cos(theta0) + x[0]*np.cos(x[3:] + theta0 - x[2]) - np.cos(x[1] - x[2] + x[3:]) - x[0]*np.cos(x[1]))**2 +
                           (np.sin(theta0) + x[0]*np.sin(x[3:] + theta0 - x[2]) - np.sin(x[1] - x[2] + x[3:]) - x[0]*np.sin(x[1]))**2,
                           np.array([np.sum(np.sin(x[3:])), np.sum(np.cos(x[3:])), x[3]])])


def jacobian_matrix_objective_function(x, theta0):
    """ Define Jacobian and add non zero entries.

    x (list): [R, Theta, Phi, psi_1, ..., psi_N],
              where Z = R e^(i Theta), w_j = e^(i psi_j) and Phi = Theta - phi.
    theta0 (ndarray): Initial conditions of the oscillators.

    """
    N = len(theta0)
    d = len(x)  # = N + 3
    dfdx = np.zeros((d, d))
    cos_term = np.cos(theta0) + x[0]*np.cos(x[3:] + theta0 - x[2]) - np.cos(x[1] - x[2] + x[3:]) - x[0]*np.cos(x[1])
    sin_term = np.sin(theta0) + x[0]*np.sin(x[3:] + theta0 - x[2]) - np.sin(x[1] - x[2] + x[3:]) - x[0]*np.sin(x[1])

    dfdx[0, :N] = 2*cos_term*(np.cos(x[3:] + theta0 - x[2]) - np.cos(x[1])) \
                  + 2*sin_term*(np.sin(x[3:] + theta0 - x[2]) - np.sin(x[1]))
    dfdx[1, :N] = 2*cos_term*(np.sin(x[1] - x[2] + x[3:]) + x[0]*np.sin(x[1])) \
                  + 2*sin_term*(-np.cos(x[1] - x[2] + x[3:]) - x[0]*np.cos(x[1]))
    dfdx[2, :N] = 2*cos_term*(x[0]*np.sin(x[3:] + theta0 - x[2]) - np.sin(x[1] - x[2] + x[3:]))\
                  + 2*sin_term*(-x[0]*np.cos(x[3:] + theta0 - x[2]) + np.cos(x[1] - x[2] + x[3:]))
    dfdx[3:, :N] = np.diag(2 * cos_term * (-x[0]*np.sin(x[3:] + theta0 - x[2]) + np.sin(x[1] - x[2] + x[3:]))
                           + 2 * sin_term * (x[0]*np.cos(x[3:] + theta0 - x[2]) - np.cos(x[1] - x[2] + x[3:])))

    dfdx[3:, N] = np.cos(x[3:])
    dfdx[3:, N+1] = -np.sin(x[3:])
    dfdx[3, N+2] = 1

    return dfdx


def get_watanabe_strogatz_initial_conditions(theta0, dispersed_guess=False, nb_guess=10, tol=1e-10):
    """ Find an appropriate set of Watanabe-Strogatz initial conditions and w constants
    by finding the roots of the objective_function_init_cond vector function.

    Args:
        theta0 (ndarray): Initial conditions of the oscillators.
        dispersed_guess (bool): If True, initial guesses for R are restrained to 0 < R < 0.2
                                instead of 0 < R < 1. Defaults to False.
        nb_guess (int): Maximum number of sets of initial guesses for the
                        optimization process. Defaults to 1.
        tol (float): parameter 'xtol' for 'hybr' method of scipy.optimize.root.
                     Defaults to 1e-10.

    Raises:
        ValueError: The optimization did not converge to successful values such that 0 < R0 < 1.

    Returns:
        (tuple of ndarrays): Initial condition for Z, initial condition for phi and w constants.

    """
    # WARNING: Since the objective function is squared, the tolerance specified here is smaller than the actual error.
    #          Note that theta0 must not be a state of majority cluster (see Watanabe-Strogatz, Sec. 4.2.3., 1994).

    R_upper = 0.2 if dispersed_guess else 1

    for _ in range(nb_guess):
        R = np.random.uniform(0.01, R_upper)
        Theta = 2*np.pi*np.random.random()
        Phi = 2*np.pi*np.random.random()
        psi = 2*np.pi*np.random.random(len(theta0))
        initial_guess = np.concatenate([np.array([R, Theta, Phi]), psi])
        solution = root(objective_function_init_cond, initial_guess, jac=jacobian_matrix_objective_function,
                        args=(theta0,), method='hybr', options={'xtol': tol, 'col_deriv': True})
        if solution.success:
            if 0 < solution.x[0] < 1:
                break

    if not solution.success:
        raise ValueError("The optimization did not converge to successful values such that 0 < R0 < 1.")

    R0, Theta0, Phi0 = solution.x[0], solution.x[1], solution.x[2]
    w = np.exp(1j*solution.x[3:])

    return R0*np.exp(1j*Theta0), Theta0 - Phi0, w
