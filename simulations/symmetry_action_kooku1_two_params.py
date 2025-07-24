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
probabilities_crossratio = np.array([[1, 0.8, 0],
                                     [1, 0., 0.8]])
probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
weights_monomial = np.array([1])  # np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
weights_crossratio = np.random.normal(1, 1, (len(sizes_crossratio), N))
weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                            probabilities=probabilities_dict, weights=weights_dict)
coupling = 1
print("W = \n", np.round(W, 3))

""" Get phase-lag matrix """
probabilities_monomial2 = np.array([0])
probabilities_crossratio2 = np.array([[1, 0.9, 0],
                                      [1, 0., 0.7]])
# np.ones((c, N)))
probabilities_nonintegrable2 = []
probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
phaselags_monomial = np.random.normal(0, 0.1, (sum(sizes_monomial), sum(sizes_monomial)))  # np.zeros((sum(sizes_monomial), sum(sizes_monomial)))
phaselags_crossratio = np.random.normal(0, 0.1, (len(sizes_crossratio), N))  # np.zeros((len(sizes_crossratio), N))
phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
alpha, chi = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                     probabilities=probabilities_dict2, phaselags=phaselags_dict)
print("alpha = \n", np.round(alpha, 3))

""" calA """
cal_A = calA(coupling, C, chi)
print("calA = \n", np.round(cal_A, 3))


""" Natural frequencies"""
omega = random_gaussian_frequencies_pintegrable(m, c, sizes, cal_A, 1, 1)
# omega1, omega2, omega3 = np.random.uniform(-1, 5, size=3)
# omega = [omega1,
#          omega2, omega2 + 2*np.imag(cal_A[0, 2] - cal_A[0, 1]), omega2 + 2*np.imag(cal_A[0, 3] - cal_A[0, 1]), omega2 + 2*np.imag(cal_A[0, 4] - cal_A[0, 1]),
#          omega3, omega3 + 2*np.imag(cal_A[1, 6] - cal_A[1, 5]), omega3 + 2*np.imag(cal_A[1, 7] - cal_A[1, 5]), omega3 + 2*np.imag(cal_A[1, 8] - cal_A[1, 5]), omega3 + 2*np.imag(cal_A[1, 9] - cal_A[1, 5])]
Omega1 = omega[1] - 2*np.imag(cal_A[0, 1])
Omega2 = omega[5] - 2*np.imag(cal_A[1, 5])
print("omega = ", omega)

""" Integration parameters """
t0, t1, dt = 0, 10, 0.01
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
Z0, phi0 = 0, 0
epsilon0 = 0
epsilon1 = 0.4
epsilon2 = 0.2
theta_transformed_1 = []
theta_transformed_2 = []
for t in tqdm(range(len(timelist))):
    zs_t = np.exp(1j*theta[t, 0])

    # First peripheral oscillators
    z1_t = np.exp(1j*theta[t, 1:1+sizes_crossratio[0]])
    args_ws1 = (z1_t, cal_A[0, 0], cal_A[0, 1:1+sizes_crossratio[0]], Omega1 - omega[0], zs_t)
    solution1 = solve_ivp(ode_symmetry_action_calS, [epsilon0, epsilon1],
                          np.array([Z0, phi0], dtype=complex), vectorized=True,
                          args=args_ws1, rtol=1e-08, atol=1e-10)
    Z1, phi1 = solution1.y[0, :], solution1.y[1, :]
    if plot_Zphi:
        plt.plot(np.real(Z1), np.imag(Z1))
        plt.show()
    theta_transformed_1.append(np.angle(disk_automorphism_bounded(Z1[-1], phi1[-1], z1_t)))

    # Second peripheral oscillators
    z2_t = np.exp(1j*theta[t, 1+sizes_crossratio[0]:])
    args_ws2 = (z2_t, cal_A[1, 0], cal_A[1, 1+sizes_crossratio[0]:], Omega2 - omega[0], zs_t)
    solution2 = solve_ivp(ode_symmetry_action_calS, [epsilon0, epsilon2],
                          np.array([Z0, phi0], dtype=complex), vectorized=True,
                          args=args_ws2, rtol=1e-08, atol=1e-10)
    Z2, phi2 = solution2.y[0, :], solution2.y[1, :]
    if plot_Zphi:
        plt.plot(np.real(Z2), np.imag(Z2))
        plt.show()
    theta_transformed_2.append(np.angle(disk_automorphism_bounded(Z2[-1], phi2[-1], z2_t)))

theta_transformed_1 = np.array(theta_transformed_1)
theta_transformed_2 = np.array(theta_transformed_2)

theta_ws1 = np.where(theta_transformed_1 < 0, 2*np.pi + theta_transformed_1, theta_transformed_1)
theta_ws2 = np.where(theta_transformed_2 < 0, 2*np.pi + theta_transformed_2, theta_transformed_2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 5))
ax1.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[2], label="Source")
ax1.set_ylabel("Phase")   # $\\theta_1(t), ..., \\theta_N(t)$")
ax1.set_ylim([-0.05, 2*np.pi + 0.05])
ax1.legend()

theta0_transformed = np.concatenate([np.array([theta0[0]]), theta_ws1[0, :], theta_ws2[0, :]])
theta_verif = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0_transformed, *args_dynamics))
theta_verif = np.where(theta_verif < 0, 2*np.pi + theta_verif, theta_verif)

ax2.plot(timelist, theta[:, 1] % (2*np.pi), color=deep[0], label="Periphery 1")
ax2.plot(timelist, theta[:, 2:1+sizes_crossratio[0]] % (2*np.pi), color=deep[0])
ax2.plot(timelist, theta_verif[:, 1] % (2*np.pi), color=deep[7], label="True")
ax2.plot(timelist, theta_verif[:, 2:1+sizes_crossratio[0]] % (2*np.pi), color=deep[7])
ax2.plot(timelist, theta_ws1[:, 0] % (2*np.pi), color=deep[1], linestyle="--", label="Theory (transf. periphery 1)")
ax2.plot(timelist, theta_ws1[:, 1:] % (2*np.pi), color=deep[1], linestyle="--")
ax2.set_ylim([-0.05, 2*np.pi + 0.05])
ax2.set_title(f"$\\epsilon_1 =$ {epsilon1}")
ax2.set_ylabel("Phase")
ax2.legend(loc=1)

ax3.plot(timelist, theta[:, 1+sizes_crossratio[0]] % (2*np.pi), color=deep[0], label="Periphery 2")
ax3.plot(timelist, theta[:, 2+sizes_crossratio[0]:] % (2*np.pi), color=deep[0])
ax3.plot(timelist, theta_verif[:, 1+sizes_crossratio[0]] % (2*np.pi), color=deep[7], label="True")
ax3.plot(timelist, theta_verif[:, 2+sizes_crossratio[0]:] % (2*np.pi), color=deep[7])
ax3.plot(timelist, theta_ws2[:, 0] % (2*np.pi), color=deep[3], linestyle="--", label="Theory (transf. periphery 2)")
ax3.plot(timelist, theta_ws2[:, 1:] % (2*np.pi), color=deep[3], linestyle="--")
ax3.set_ylim([-0.05, 2*np.pi + 0.05])
ax3.set_title(f"$\\epsilon_2 =$ {epsilon2}")
ax3.set_ylabel("Phase")

ax3.set_xlabel("Time $t$")
ax3.legend(loc=1)
plt.show()