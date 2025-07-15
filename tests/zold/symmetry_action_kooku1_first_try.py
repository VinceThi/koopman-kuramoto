import numpy as np
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto
from dynamics.symmetries import ode_symmetry_action_calS, disk_automorphism_bounded, inverse_disk_automorphism
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from plots.config_rcparams import *


""" Dynamical parameters """
# Weight matrix and coupling
coupling = 1
w1, w2, w3, w4, w5 = 1., 0., 1., 0., 1.   #  1., 0., -0.598, 0., 1.512
W = np.array([[0., 0., 0., 0., 0.],
              [w1, 0., w3, w4, w5],
              [w1, w2, 0., w4, w5],
              [w1, w2, w3, 0., w5],
              [w1, w2, w3, w4, 0.]])
print("W = \n", W)

# Phase lags
a1, a2, a3, a4, a5 = 0., 0., 0., 0., 0.   #   0.1, -0.7, np.pi/2-0.1, 0.9, 1.  #
alpha =  np.array([[0., 0., 0., 0., 0.],
                   [a1, 0., a3, a4, a5],
                   [a1, a2, 0., a4, a5],
                   [a1, a2, a3, 0., a5],
                   [a1, a2, a3, a4, 0.]])
print("alpha = \n", alpha)

# Natural frequencies
calA = coupling/2*np.array([[w1*np.exp(1j*a1), w2*np.exp(1j*a2), w3*np.exp(1j*a3),
                             w4*np.exp(1j*a4), w5*np.exp(1j*a5)]])
omega1, omega2 = 2.1345, 0.493341
omega =  [omega1, omega2, omega2 + 2*np.imag(calA[0, 2] - calA[0, 1]),
          omega2 + 2*np.imag(calA[0, 3] - calA[0, 1]), omega2 + 2*np.imag(calA[0, 4] - calA[0, 1])]
Omega = omega2 - 2*np.imag(calA[0, 1])

""" Integration parameters """
t0, t1, dt = 0, 100, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
time_mean_index = int((t1//dt)//3)
N = len(alpha[0])
np.random.seed(499)
theta0 = np.random.uniform(0, 2*np.pi, N)
theta0_source = theta0[0]

""" Integrate complete dynamics """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
theta = np.where(theta < 0, 2*np.pi + theta, theta)

""" Integrate reduced dynamics """
Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0[1:], N-1, nb_guess=5000)
xis = np.exp(1j*theta0_source)  # TODO not sure
args_ws = (w, calA[0], Omega, xis)
solution = np.array(integrate_dopri45(t0, t1, dt, ode_symmetry_action_calS, np.array([Z0, phi0]), *args_ws))
Z, phi = solution[:, 0], solution[:, 1]

plt.plot(np.real(Z), np.imag(Z))
plt.plot(phi)
plt.show()

epsilon = 100  # Index
theta_transformed = []
for t in range(len(timelist)):
    z_t = np.exp(theta[t, 1:])
    w_t = inverse_disk_automorphism(Z0, phi0, z_t)  # TODO not sure
    # Apply transformation at each time
    theta_transformed.append(np.angle(disk_automorphism_bounded(Z[epsilon], phi[epsilon], w_t)))
theta_transformed = np.array(theta_transformed)
theta_ws = np.where(theta_transformed < 0, 2*np.pi + theta_transformed, theta_transformed)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
ax1.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[2], label="Source")
ax1.plot(timelist, theta[:, 1:] % (2*np.pi), color=deep[0], label="Periphery")
ax2.set_ylabel("Solutions")   # $\\theta_1(t), ..., \\theta_N(t)$")
plt.legend()

ax2.plot(timelist, theta_ws % (2*np.pi), color=deep[1], label="Transformed solution")
plt.ylabel("Transformed solutions")   # $\\theta_1(t), ..., \\theta_N(t)$")
plt.xlabel("Time $t$")
# plt.legend()
plt.show()