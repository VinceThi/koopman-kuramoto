# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi
from scipy.integrate import solve_ivp


N = 4

""" Dynamical parameters """
t0, t1, dt = 0, 6, 0.005
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 1
coupling = 1
W = np.ones((N, N))
theta0 = np.random.uniform(0, 2*np.pi, N)

""" Integrate Kuramoto dynamics """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))


""" Integrate Kuramoto dynamics under the change of variable """


def kuramoto_dwdt(t, w, y, current_index, coupling, omega):
    w1, w2, w3, w4 = w
    Y = np.exp(1j*y[current_index])
    p = coupling/2*(Y + w2 + w3 + ((w3 - Y)*w1*w2 - (w3 - w2)*Y)/((w3 - Y)*w1 - (w3 - w2)))
    pbar = np.conj(p)
    dw1dt = 0
    dw2dt = p + 1j*omega*w2 - pbar*w2**2
    dw3dt = p + 1j*omega*w3 - pbar*w3**2
    dw4dt = w1
    return np.array([dw1dt, dw2dt, dw3dt, dw4dt])


args_dyn = (coupling, omega, )
y = theta[:, 0]
z0 = np.exp(1j*theta0)
w10 = ((z0[2] - z0[1])*(z0[3] - z0[0]))/((z0[2] - z0[0])*(z0[3] - z0[1]))
w0 = np.array([w10, z0[1], z0[2], w10*t0])
#                                  = 0
w = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, kuramoto_dwdt, w0, y, *args_dyn))


# """ Integrate Kuramoto dynamics under the change of variable """
# def kuramoto_change_var(y, w, coupling, omega):
#     w1, w2, w3, w4 = w
#     Y = np.exp(1j*y)
#     Ybar = np.exp(-1j*y)
#     p = coupling/2*(Y + w2 + w3 + ((w3 - Y)*w1*w2 - (w3 - w2)*Y)/((w3 - Y)*w1 - (w3 - w2)))
#     pbar = np.conj(p)
#     dw1dy = 0
#     dw2dy = (p + 1j*omega*w2 - pbar*w2**2)/(omega + 1j*(pbar*Y - p*Ybar))
#     dw3dy = (p + 1j*omega*w3 - pbar*w3**2)/(omega + 1j*(pbar*Y - p*Ybar))
#     dw4dy = w1/(omega + 1j*(pbar*Y - p*Ybar))
#     return np.array([dw1dy, dw2dy, dw3dy, dw4dy])
#
#
# args_dyn = (coupling, omega, )
# y0, y1, dy = theta0[0], theta[-1, 0], 0.001
# ylist = np.linspace(y0, y1, int(y1 / dy))
# z0 = np.exp(1j*theta0)
# w10 = ((z0[2] - z0[1])*(z0[3] - z0[0]))/((z0[2] - z0[0])*(z0[3] - z0[1]))
# w0 = np.array([w10, z0[1], z0[2], 0])
# # w_sol = np.array(integrate_dopri45(y0, y1, dy, kuramoto_change_var, w0, *args_dyn))
# sol = solve_ivp(kuramoto_change_var, [y0, y1], w0, t_eval=np.linspace(y0, y1, 100), args=args_dyn)
# w_sol = sol.y
# w_solR = np.real(w_sol)
# w_solI = np.imag(w_sol)
# y_sol = sol.t
# print(np.angle(sol.y), np.abs(sol.y))

plt.subplot(211)
for i in range(N):
    plt.plot(timelist, theta[:, i] % (2*np.pi), label=f"$\\theta_{i+1}$")
plt.legend()

plt.subplot(212)
plt.plot(timelist, np.real(w[:, 0]), label="$w_1$")  # Integral of motion
plt.plot(timelist, np.angle(w[:, 1]) % (2*np.pi), label="$w_2$")  # w_2 on the unit circle
plt.plot(timelist, np.angle(w[:, 2]) % (2*np.pi), label="$w_3$")  # w_3 on the unit circle
plt.plot(timelist, np.real(w[:, 3]), label="$w_4$")  # Real
# for j in range(N):
#     plt.plot(y_sol, w_solR[j, :], label=f"Re($w_{j+1}$)")
#     plt.plot(y_sol, w_solI[j, :], label=f"Im($w_{j+1}$)")
# plt.plot(y_sol, np.real(w_solI[0, :]))   # Integral of motion
# plt.plot(y_sol, np.angle(w_solI[1, :]))  # They lie (up to precision of the integrator), on the unit circle
# plt.plot(y_sol, np.angle(w_solI[2, :]))  # They lie (up to precision of the integrator), on the unit circle
# plt.plot(y_sol, np.abs(w_solI[3, :]))    # The phase is null
plt.legend()
plt.xlabel("Time $t$")
# plt.xlabel("New independent variable $y$")
plt.show()
