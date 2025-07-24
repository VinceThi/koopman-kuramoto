import numpy as np
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto
from dynamics.symmetries import ode_symmetry_action_calS, disk_automorphism_bounded
from plots.config_rcparams import *
from tqdm import tqdm
from scipy.integrate import solve_ivp


plot_Zphi = False


sizes_monomial = np.array([1], dtype=int)
sizes_crossratio = np.array([4, 5], dtype=int)
size_nonintegrable = np.array([], dtype=int)
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])

""" Dynamical parameters """
# Weight matrix and coupling
coupling = 1
binary_vector = (np.random.rand(11, ) < np.array([1, 1, 0.4, 0.5, 0.3, 0.9, 0.5, 0.9, 0.8, 0.4, 0.7])).astype(int)
weight_vector = np.random.normal(0.1, 1, (11, ))
w = binary_vector*weight_vector
w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10]

W = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [w0, 0., w3, w4, w5, 0., 0., 0., 0., 0.],
              [w0, w2, 0., w4, w5, 0., 0., 0., 0., 0.],
              [w0, w2, w3, 0., w5, 0., 0., 0., 0., 0.],
              [w0, w2, w3, w4, 0., 0., 0., 0., 0., 0.],
              [w1, 0., 0., 0., 0., 0., w7, w8, w9, w10],
              [w1, 0., 0., 0., 0., w6, 0., w8, w9, w10],
              [w1, 0., 0., 0., 0., w6, w7, 0., w9, w10],
              [w1, 0., 0., 0., 0., w6, w7, w8, 0., w10],
              [w1, 0., 0., 0., 0., w6, w7, w8, w9, 0.]])
print("W = \n", np.round(W, 3))

# Phase lags
binary2_vector = (np.random.rand(11, ) < np.array([0.8, 0.5, 0.7, 0.6, 0.8, 0.9, 0.2, 0.5, 0.6, 0.7, 0.7])).astype(int)
phaselag_vector = np.random.normal(1, 1, (11, ))
a = binary2_vector*phaselag_vector
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]
alpha = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [a0, 0., a3, a4, a5, 0., 0., 0., 0., 0.],
                  [a0, a2, 0., a4, a5, 0., 0., 0., 0., 0.],
                  [a0, a2, a3, 0., a5, 0., 0., 0., 0., 0.],
                  [a0, a2, a3, a4, 0., 0., 0., 0., 0., 0.],
                  [a1, 0., 0., 0., 0., 0., a7, a8, a9, a10],
                  [a1, 0., 0., 0., 0., a6, 0., a8, a9, a10],
                  [a1, 0., 0., 0., 0., a6, a7, 0., a9, a10],
                  [a1, 0., 0., 0., 0., a6, a7, a8, 0., a10],
                  [a1, 0., 0., 0., 0., a6, a7, a8, a9, 0.]])
print("alpha = \n", np.round(alpha, 3))

# Natural frequencies
cal_A = coupling/2*np.array([[w0*np.exp(-1j*a0), w2*np.exp(-1j*a2), w3*np.exp(-1j*a3),
                            w4*np.exp(-1j*a4), w5*np.exp(-1j*a5), 0., 0., 0., 0., 0.],
                            [w1*np.exp(-1j*a1),  0., 0., 0., 0., w6*np.exp(-1j*a6), w7*np.exp(-1j*a7),
                            w8*np.exp(-1j*a8), w9*np.exp(-1j*a9), w10*np.exp(-1j*a10)]])
omega1, omega2, omega3 = np.random.uniform(-1, 5, size=3)
omega = [omega1,
         omega2, omega2 + 2*np.imag(cal_A[0, 2] - cal_A[0, 1]), omega2 + 2*np.imag(cal_A[0, 3] - cal_A[0, 1]), omega2 + 2*np.imag(cal_A[0, 4] - cal_A[0, 1]),
         omega3, omega3 + 2*np.imag(cal_A[1, 6] - cal_A[1, 5]), omega3 + 2*np.imag(cal_A[1, 7] - cal_A[1, 5]), omega3 + 2*np.imag(cal_A[1, 8] - cal_A[1, 5]), omega3 + 2*np.imag(cal_A[1, 9] - cal_A[1, 5])]
Omega1 = omega2 - 2*np.imag(cal_A[0, 1])
Omega2 = omega3 - 2*np.imag(cal_A[1, 5])
print("omega = ", f"{omega1, omega2, omega3}")

""" Integration parameters """
t0, t1, dt = 0, 10, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))
N = len(alpha[0])
np.random.seed(12)
theta0 = np.random.uniform(0, 2*np.pi, N)
theta0_source = theta0[0]

""" Integrate complete dynamics """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
theta = np.where(theta < 0, 2*np.pi + theta, theta)


""" Integrate symmetry """
Z0, phi0 = 0, 0
epsilon0 = 0
epsilon1 = 0.2
epsilon2 = 0.4
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
