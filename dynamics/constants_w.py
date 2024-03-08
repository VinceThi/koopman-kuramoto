import numpy as np
from scipy.optimize import root


def constraints_psi(psi, cross_ratios):
    """ Define the vector function to pass to scipy.optimize.root from the set of constraints
    psi: phase of w
    """
    output = []
    # add N-3 cross-ratio constraints
    for i, cr in enumerate(cross_ratios[:-2]):
        wa, wb, wc, wd = np.exp(1j * psi[i]), np.exp(1j * psi[i+1]), np.exp(1j * psi[i+2]), np.exp(1j * psi[i+3])
        new_constraint = (1-cr)*wc*wd - wc*wb - wa*wd + (1-cr)*wa*wb + cr*wc*wa + cr*wb*wd
        new_constraint_real = np.real(new_constraint)
        output.append(new_constraint_real)
    wa, wb, wc, wd = np.exp(1j * psi[-5]), np.exp(1j * psi[-4]), np.exp(1j * psi[-3]), np.exp(1j * psi[-2])
    last_constraint = (1-cr)*wc*wd - wc*wb - wa*wd + (1-cr)*wa*wb + cr*wc*wa + cr*wb*wd
    last_constraint_real = np.real(last_constraint)
    last_constraint_imag = np.imag(last_constraint)
    output.append(last_constraint_real)
    output.append(last_constraint_imag)

    # add the 2 constraints ensuring that |Z| = 0 can be interpreted as incoherence (see Watanabe, Strogatz, 1994)
    output.append(np.sum(np.sin(psi)))
    output.append(np.sum(np.cos(psi)))

    # add a last arbitrary constraint
    output.append(np.sum(psi))

    return np.array(output)


def get_w(cross_ratios, N, nb_iter=500):
    """ Find the solutions for w using different sets of random initial values for the constants and
    the N constraints defined in constraints_psi """
    for i in range(nb_iter):
        init_psi = 2*np.pi*np.random.random(N)
        solution = root(constraints_psi, init_psi, args=(cross_ratios,), method='hybr', tol=1e-10)
        if solution.success:
            break
    if not solution.success:
        raise ValueError("The optimization did not converge to successful values of w.")
    return np.exp(1j*solution.x)
