# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.symmetries import ode_symmetry_action_calS, disk_automorphism_bounded, inverse_disk_automorphism
from tqdm import tqdm
from scipy.integrate import solve_ivp
from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45

plot_Zphi = False

""" Partition parameters"""
sizes_monomial = np.array([1], dtype=int)
sizes_crossratio = np.array([4, 5], dtype=int)
size_nonintegrable = np.array([], dtype=int)
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
q = len(sizes)  # Number of parts
N = np.sum(sizes, dtype=int)
m = len(sizes_monomial)
c = len(sizes_crossratio)

""" Get weight matrix """
random_exponents = np.array([1])  # np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
probabilities_monomial = np.array([0])
probabilities_crossratio = np.array([[1, 0.5, 0.5, 0.5, 0.5, 0., 0., 0., 0., 0.],
                                     [1, 0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5]])
# np.random.rand(c, N)
probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
weights_monomial = np.array([1])  # np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
weights_crossratio = np.random.normal(1, 1, (len(sizes_crossratio), N))
# np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# np.random.normal(1, 1, (len(sizes_crossratio), N))
weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

W = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                         probabilities=probabilities_dict, weights=weights_dict)
coupling = 1
print("W = \n", W)

""" Get phase-lag matrix """
probabilities_monomial2 = np.array([0])
probabilities_crossratio2 = np.ones((c, N))
probabilities_nonintegrable2 = []
probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
phaselags_monomial = np.random.normal(0, 0.1, (sum(sizes_monomial), sum(sizes_monomial))) # np.zeros((sum(sizes_monomial), sum(sizes_monomial)))
phaselags_crossratio = np.random.normal(0, 0.1, (len(sizes_crossratio), N)) # np.zeros((len(sizes_crossratio), N))
phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
alpha = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                probabilities=probabilities_dict2, phaselags=phaselags_dict)
cal_A = calA(coupling, weights_crossratio, phaselags_crossratio)
print("alpha = \n", alpha)

""" Natural frequencies"""
omega = random_gaussian_frequencies_pintegrable(c, sizes, cal_A, 1, 1)
Omega1 = omega[1] - 2*np.imag(calA[0, 1])
Omega2 = omega[5] - 2*np.imag(calA[1, 6])
print("omega = ", omega)

""" Integration parameters """
t0, t1, dt = 0, 20, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))
N = len(alpha[0])
# np.random.seed(12)
theta0 = np.random.uniform(0, 2*np.pi, N)
theta0_source = theta0[0]

""" Integrate complete dynamics """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
theta = np.where(theta < 0, 2*np.pi + theta, theta)


""" Integrate symmetry """
epsilon0 = 0
epsilon1 = 1  # Index
epsilon2 = 1
theta_transformed = []
for t in tqdm(range(len(timelist))):
    zs_t = np.exp(1j*theta[t, 0])
    z_t = np.exp(1j*theta[t, 1:])
    Z0, phi0 = 0, 0
    args_ws = (z_t, calA, Omega - omega1, zs_t)
    solution = solve_ivp(ode_symmetry_action_calS, [epsilon0, epsilon],
                         np.array([Z0, phi0], dtype=complex), vectorized=True,
                         args=args_ws, rtol=1e-08, atol=1e-10)
    Z, phi = solution.y[0, :], solution.y[1, :]
    if plot_Zphi:
        plt.plot(np.real(Z), np.imag(Z))
        plt.show()
    theta_transformed.append(np.angle(disk_automorphism_bounded(Z[-1], phi[-1], z_t)))


theta_transformed = np.array(theta_transformed)

theta_ws = np.where(theta_transformed < 0, 2*np.pi + theta_transformed, theta_transformed)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5))
ax1.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[2], label="Source")
ax1.plot(timelist, theta[:, 1:] % (2*np.pi), color=deep[0], label="Periphery")
ax1.set_ylabel("Solutions")   # $\\theta_1(t), ..., \\theta_N(t)$")
ax1.set_ylim([-0.05, 2*np.pi + 0.05])
plt.legend()

theta_verif = np.array(integrate_dopri45(t0, t1, dt, kuramoto,
                                         np.concatenate([np.array([theta0[0]]), theta_ws[0, :]]), *args_dynamics))
theta_verif = np.where(theta_verif < 0, 2*np.pi + theta_verif, theta_verif)
ax2.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[2], label="Source")
ax2.plot(timelist, theta_verif[:, 1] % (2*np.pi), color=deep[7], label="True")
ax2.plot(timelist, theta_verif[:, 2:] % (2*np.pi), color=deep[7])
ax2.plot(timelist, theta_ws[:, 0] % (2*np.pi), color=deep[1], linestyle="--", label="Theory")
ax2.plot(timelist, theta_ws[:, 1:] % (2*np.pi), color=deep[1], linestyle="--")
ax2.set_ylim([-0.05, 2*np.pi + 0.05])
ax2.set_title(f"$\\epsilon =$ {epsilon}")
ax2.set_ylabel("Transformed solutions")   # $\\theta_1(t), ..., \\theta_N(t)$")
plt.xlabel("Time $t$")
plt.legend()
plt.show()


""" Old code """
# zs_t = np.exp(1j*theta[t, 0])
# print("\n", "theta(t) = ", theta[t, 1:])
# Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta[t, 1:], dispersed_guess=False, nb_guess=5000)
# args_ws = (w, calA[0], Omega, zs_t)
# solution = np.array(integrate_dopri45(epsilon0, epsilon, depsilon, ode_symmetry_action_calS,
#                                       np.array([Z0, phi0]), *args_ws))
# Z, phi = solution[:, 0], solution[:, 1]
# print("w(t) = ",  np.round(w, 3))
# theta_transformed.append(np.angle(disk_automorphism_bounded(Z[-1], phi[-1], w)))
# print(disk_automorphism_bounded(Z[-1], phi[-1], w))

# solution = np.array(integrate_dopri45(epsilon0, epsilon, depsilon, ode_symmetry_action_calS,
#                                       np.array([Z0, phi0]), *args_ws))
# Z, phi = solution[:, 0], solution[:, 1]
#