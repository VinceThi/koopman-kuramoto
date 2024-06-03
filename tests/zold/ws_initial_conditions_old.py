import numpy as np
from scipy.optimize import root
from tqdm import tqdm


def objective_function_init_cond(x, theta0):
    """ R, Theta, Phi, psi_1, ..., psi_N = x
     where Z = R e^(i Theta), w_j = e^(i psi_j), Phi = Theta - phi """
    return np.concatenate([np.tan((theta0 - x[1])/2) - (1 - x[0])/(1 + x[0])*np.tan((x[3:] - x[2])/2),
                           np.array([np.sum(np.sin(x[3:])), np.sum(np.cos(x[3:])), x[3]])])


def jacobian_matrix_objective_function(x, theta0):
    """ Define Jacobian and add non zero entries """
    N = len(theta0)
    d = len(x)  # = N + 3
    dfdx = np.zeros((d, d))

    dfdx[0, :N] = 2*np.tan((x[3:] - x[2])/2)/(1 + x[0])**2
    dfdx[1, :N] = -1/(2*np.cos((theta0 - x[1])/2)**2)
    dfdx[2, :N] = (1 - x[0])/(2*(1 + x[0])*np.cos((x[3:] - x[2])/2)**2)
    dfdx[3:, :N] = np.diag((x[0] - 1)/(2*(1 + x[0])*np.cos((x[3:] - x[2])/2)**2))

    dfdx[3:, N] = np.cos(x[3:])
    dfdx[3:, N+1] = -np.sin(x[3:])
    dfdx[3, N+2] = 1

    return dfdx


def get_watanabe_strogatz_initial_conditions(theta0, N, nb_guess=10000, tol=1e-10, tol_ws=1e-6):
    """ Note that theta0 must not be a state of majority cluster (see Watanabe-Strogatz, Sec. 4.2.3., 1994). """

    for _ in tqdm(range(nb_guess)):
        R = np.random.uniform(0, 1)
        Theta = 2*np.pi*np.random.random()
        Phi = 2*np.pi*np.random.random()
        psi = 2*np.pi*np.random.random(N)
        initial_guess = np.concatenate([np.array([R, Theta, Phi]), psi])
        solution = root(objective_function_init_cond, initial_guess, jac=jacobian_matrix_objective_function,
                        args=(theta0,), method='hybr', options={'xtol': tol, 'col_deriv': True})
        R0, Theta0, Phi0 = solution.x[0], solution.x[1], solution.x[2]
        if solution.success:
            if 0 < solution.x[0] < 1:
                if np.all(np.abs(np.tan((theta0-Theta0)/2) - (1-R0)/(1+R0)*np.tan((solution.x[3:]-Phi0)/2)) < tol_ws):
                    break

    if not solution.success:
        raise ValueError("The optimization did not converge to successful values such that 0 < R0 < 1.")

    R0, Theta0, Phi0 = solution.x[0], solution.x[1], solution.x[2]
    w = np.exp(1j*solution.x[3:])

    return R0*np.exp(1j*Theta0), Theta0 - Phi0, w
