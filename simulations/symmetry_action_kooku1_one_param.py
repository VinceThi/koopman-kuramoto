import numpy as np
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto
from dynamics.symmetries import ode_symmetry_action_calS, disk_automorphism_bounded, inverse_disk_automorphism
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from plots.config_rcparams import *
from tqdm import tqdm
from scipy.integrate import solve_ivp


plot_Zphi = False


""" Dynamical parameters """
# Weight matrix and coupling
coupling = 1
binary_vector = (np.random.rand(5, ) < np.array([1, 0.4, 0.5, 0.3, 0.9])).astype(int)
weight_vector = np.random.normal(1, 1, (5, ))
w = binary_vector*weight_vector
w1, w2, w3, w4, w5 = 1., 0., 1., 0., 1.     # w[0], w[1], w[2], w[3], w[4]  # 1, 1., -0.598, 0.1, 1.512      #
W = np.array([[0., 0., 0., 0., 0.],
              [w1, 0., w3, w4, w5],
              [w1, w2, 0., w4, w5],
              [w1, w2, w3, 0., w5],
              [w1, w2, w3, w4, 0.]])
print("W = \n", W)

# Phase lags
binary2_vector = (np.random.rand(5, ) < np.array([0.8, 0.5, 0.7, 0.6, 0.8])).astype(int)
phaselag_vector = np.random.normal(1, 1, (5, ))
a = binary2_vector*phaselag_vector
a1, a2, a3, a4, a5 = a[0], a[1], a[2], a[3], a[4]    # 0.1, -0.7, np.pi/2-0.1, 0.9, 1.     # 0., 0., 0., 0., 0. #
alpha = np.array([[0., 0., 0., 0., 0.],
                  [a1, 0., a3, a4, a5],
                  [a1, a2, 0., a4, a5],
                  [a1, a2, a3, 0., a5],
                  [a1, a2, a3, a4, 0.]])
print("alpha = \n", alpha)

# Natural frequencies
calA = coupling/2*np.array([w1*np.exp(-1j*a1), w2*np.exp(-1j*a2), w3*np.exp(-1j*a3),
                            w4*np.exp(-1j*a4), w5*np.exp(-1j*a5)])
omega1, omega2 = np.random.uniform(-1, 5), np.random.uniform(-1, 5)
omega = [omega1, omega2, omega2 + 2*np.imag(calA[2] - calA[1]),
         omega2 + 2*np.imag(calA[3] - calA[1]), omega2 + 2*np.imag(calA[4] - calA[1])]
Omega = omega2 - 2*np.imag(calA[1])
print("omega = ", f"{omega1, omega2}")

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
epsilon = 1  # Index
# depsilon = 0.01
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