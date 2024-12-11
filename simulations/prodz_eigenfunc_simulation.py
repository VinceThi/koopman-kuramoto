# Script for simulating kuramoto system with an eigenfunction in the form of a monomial of z variables

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.integrate import integrate_dopri45


# set dynamical parameters
N = 3
ones_vec = np.ones((N, 1))
w_coeffs = np.array([[1., 1., 1.]])
w_matrix = ones_vec @ w_coeffs
np.fill_diagonal(w_matrix, 0)
print(w_matrix)
omega = np.array([1] * N)
coupling = 0.5
alpha = 0

# integrate the dynamics
x0 = 2*np.pi*np.random.random(N)
print(x0)
t0, t1, dt = 0, 15, 0.001
time = np.linspace(t0, t1, int(t1 / dt))

args_dynamics = (w_matrix, coupling, omega, alpha)
solution = solve_ivp(kuramoto_sakaguchi, (t0, t1), x0, method='DOP853', args=args_dynamics,
                 t_eval=time, atol=1e-10, rtol=1e-10)
print('success', solution.success)
theta = solution.y.T

k1 = 1/(np.sum(w_coeffs))
# k2 = 2
mu1 = k1 * w_coeffs
# mu2 = k2 * w_coeffs
eigenfunc1 = mu1 @ theta.T
# eigenfunc2 = mu2 @ theta.T

plt.figure()
for j in range(0, N):
    plt.plot(time, (theta[:, j]) % (2*np.pi), linewidth=0.5)
plt.plot(time, eigenfunc1[0] % (2*np.pi), linewidth=1, label='eigenfunc1')
plt.plot(time, (eigenfunc1[0] + 2*np.pi/3) % (2*np.pi), linewidth=1, label='eigenfunc2')
plt.plot(time, (eigenfunc1[0] + 2*np.pi*2/3) % (2*np.pi), linewidth=1, label='eigenfunc3')
# plt.plot(time, (2*eigenfunc2[0]/(k2*np.sum(w_coeffs))) % (2*np.pi), linewidth=1, label='eigenfunc')
ylab = plt.ylabel('$\\theta_j$', labelpad=20)
ylab.set_rotation(0)
plt.xlabel('Time $t$')
# plt.ylim([-1, 2*np.pi + 1])
plt.tick_params(axis='both', which='major')
plt.legend()
plt.tight_layout()
plt.show()
