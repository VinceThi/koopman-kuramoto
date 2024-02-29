import numpy as np
from scipy.optimize import root
from dynamics.constants_of_motion import cross_ratio_z


# define the system parameters and the initial values of the microscopic variables
n = 100
init_thetas = np.random.random(n) * 2 * np.pi
init_z = np.exp(1j * init_thetas)
a = np.ones((n, n))

# compute the values of the independent cross-ratios from the initial values of the microscopic variables.
cross_ratios = []
for i, init_z_i in enumerate(init_z[:-3]):
    cross_ratios.append(np.real(cross_ratio_z(init_z_i, init_z[i+1], init_z[i+2], init_z[i+3])))
# print(cross_ratios)

# define the vector function to pass to scipy.optimize.root from the set of constraints
def constraints_phi(phi, cross_ratios):
    output = []
    # add N-3 cross-ratio constraints
    for i, cr in enumerate(cross_ratios[:-2]):
        wa, wb, wc, wd = np.exp(1j * phi[i]), np.exp(1j * phi[i+1]), np.exp(1j * phi[i+2]), np.exp(1j * phi[i+3])
        new_constraint = wc*wd*(1-cr) - wc*wb - wa*wd + wa*wb*(1-cr) + cr*wc*wa + cr*wb*wd
        new_constraint_real = np.real(new_constraint)
        output.append(new_constraint_real)
    wa, wb, wc, wd = np.exp(1j * phi[-5]), np.exp(1j * phi[-4]), np.exp(1j * phi[-3]), np.exp(1j * phi[-2])
    last_constraint = wc*wd*(1-cr) - wc*wb - wa*wd + wa*wb*(1-cr) + cr*wc*wa + cr*wb*wd
    last_constraint_real = np.real(last_constraint)
    last_constraint_imag = np.imag(last_constraint)
    output.append(last_constraint_real)
    output.append(last_constraint_imag)
    # add the 3 other well-chosen constraints
    output.append(np.sum(np.sin(phi)))
    output.append(np.sum(np.cos(phi)))
    output.append(np.sum(phi))
    return np.array(output)

# find the solutions using different sets of random initial values for the constants
success = False
for i in range(100):
    init_w_angles = np.random.random(n) * np.pi
    init_w = np.exp(1j * init_w_angles)
    solution = root(constraints_phi, init_w_angles, args=(cross_ratios), method='hybr')
    success = solution.success
    print(f'success {i}:', success)
    if success:
        break
print('final success:', success)
print('init_w_angles', init_w_angles)
print('success:', solution.success)
phi = solution.x
w = np.exp(1j * phi)
print('w', w)



# USELESS CODE IN CASE

# def constraints_w(w, cross_ratios):
    # output = []
    # # add N-3 cross-ratio constraints
    # for i, cr in enumerate(cross_ratios):
        # wa, wb, wc, wd = w[i], w[i+1], w[i+2], w[i+3]
        # new_constraint = wc*wd*(1-cr) - wc*wb - wa*wd + wa*wb*(1-cr) + cr*wc*wa + cr*wb*wd
        # output.append(new_constraint)
    # # add the 3 other well-chosen constraints
    # output.append(np.sum(np.real(w)))
    # output.append(np.sum(np.imag(w)))
    # output.append(np.sum(np.angle(w)))
    # return np.array(output)

# # find the solutions
# init_w_angles = np.random.random(n) * np.pi
# init_w = np.exp(1j * init_w_angles)
# print('init_w', init_w)
# solution = root(constraints, init_w, args=(cross_ratios), method='hybr')

# print(solution)
