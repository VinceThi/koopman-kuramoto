# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi
from scipy.integrate import solve_ivp


N = 5

""" Dynamical parameters """
t0, t1, dt = 0.00001, 3, 0.005
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
"""
def kuramoto_dwdt(t, w, y, current_index, coupling, omega):
    # Change of coord: y = theta1, w1 = 1/c1, w2 = z2, w3 = z3, w4 = t/c1, w5 = t/c2
    w1, w2, w3, w4, w5 = w
    Z1 = np.exp(1j*y[current_index])
    Z4 = ((w3 - Z1)*w1*w2 - (w3 - w2)*Z1)/((w3 - Z1)*w1 - (w3 - w2))
    Z5 = ((Z4 - w3)*w2*w4 - (Z4 - w2)*w1*w3*w5)/((Z4 - w3)*w4 - (Z4 - w2)*w1*w5)
    # Z5 = ((Z1 - w2)*w1*w3*w5 - (Z1 - w3)*w2*w4)/((Z1 - w2)*w1*w5 - (Z1 - w3)*w4)
    # print(Z4, Z5, (Z4 - w3)*w2*w4 - (Z4 - w2)*w1*w3*w5, (Z4 - w3)*w4 - (Z4 - w2)*w1*w5)
    # print(Z4, Z5, (Z1 - w2)*w1*w3*w5)
    p = coupling/2*(Z1 + w2 + w3 + Z4 + Z5)
    pbar = np.conj(p)
    dw1dt = 0
    dw2dt = p + 1j*omega*w2 - pbar*w2**2
    dw3dt = p + 1j*omega*w3 - pbar*w3**2
    dw4dt = w1
    dw5dt = ((Z4 - w3)*(Z5 - w2))/((Z4 - w2)*(Z5 - w3))
    return np.array([dw1dt, dw2dt, dw3dt, dw4dt, dw5dt])


args_dyn = (coupling, omega, )
y = theta[:, 0]
z0 = np.exp(1j*theta0)
w10 = ((z0[2] - z0[1])*(z0[3] - z0[0]))/((z0[2] - z0[0])*(z0[3] - z0[1]))
c20 = ((z0[3] - z0[1])*(z0[4] - z0[2]))/((z0[3] - z0[2])*(z0[4] - z0[1]))
# c20 = ((z0[0] - z0[1])*(z0[4] - z0[2]))/((z0[0] - z0[2])*(z0[4] - z0[1]))
w0 = np.array([w10, z0[1], z0[2], w10*t0, t0/c20])
w = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, kuramoto_dwdt, w0, y, *args_dyn))
"""


def kuramoto_dwdy(y, w, coupling, omega):
    """ Change of coord: y = theta1, w1 = 1/c1, w2 = z2, w3 = z3, w4 = t/c1, w5 = t/c2"""
    w1, w2, w3, w4, w5 = w
    Z1 = np.exp(1j*y)
    Z4 = ((w3 - Z1)*w1*w2 - (w3 - w2)*Z1)/((w3 - Z1)*w1 - (w3 - w2))
    Z5 = ((Z4 - w3)*w2*w4 - (Z4 - w2)*w1*w3*w5)/((Z4 - w3)*w4 - (Z4 - w2)*w1*w5)
    q = coupling/2*(Z1 + w2 + w3 + Z4 + Z5)
    qbar = np.conj(q)
    dYdt = omega + 1j*(qbar*Z1 - q*np.conj(Z1))
    dw1dy = 0
    dw2dy = (q + 1j*omega*w2 - qbar*w2**2)/dYdt
    dw3dy = (q + 1j*omega*w3 - qbar*w3**2)/dYdt
    dw4dy = w1/dYdt
    dw5dy = ((Z4 - w3)*(Z5 - w2))/((Z4 - w2)*(Z5 - w3))/dYdt
    return np.array([dw1dy, dw2dy, dw3dy, dw4dy, dw5dy])


args_dyn = (coupling, omega, )
y0, y1 = theta0[0], theta[-1, 0]
z0 = np.exp(1j*theta0)
w10 = ((z0[2] - z0[1])*(z0[3] - z0[0]))/((z0[2] - z0[0])*(z0[3] - z0[1]))
c20 = ((z0[3] - z0[1])*(z0[4] - z0[2]))/((z0[3] - z0[2])*(z0[4] - z0[1]))
w0 = np.array([w10, z0[1], z0[2], w10*t0, t0/c20])
sol = solve_ivp(kuramoto_dwdy, [y0, y1], w0, t_eval=np.linspace(y0, y1, len(timelist)), args=args_dyn)
w_sol = sol.y
# w_solR = np.real(w_sol)
# w_solI = np.imag(w_sol)
y_sol = sol.t

time = w_sol[3, :]/w_sol[0, :]    # warning: this is not a linear function, the time steps are adaptive


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
plt.ylim([-0.1, 2*np.pi + 0.1])
plt.xlabel("Time $t$")

plt.subplot(212)
# plt.plot(timelist, np.real(w[:, 0]), label="$w_1$")  # Integral of motion
# plt.plot(timelist, np.angle(w[:, 1]) % (2*np.pi), label="arg($w_2$)")  # w_2 on the unit circle
# plt.plot(timelist, np.angle(w[:, 2]) % (2*np.pi), label="arg($w_3$)")  # w_3 on the unit circle
# plt.plot(timelist, np.real(w[:, 3]), label="$w_4$")  # Real
# plt.plot(timelist, np.real(w[:, 4]), label="$w_5$")  # Real
# # for j in range(N):
# #     plt.plot(y_sol, w_solR[j, :], label=f"Re($w_{j+1}$)")
# #     plt.plot(y_sol, w_solI[j, :], label=f"Im($w_{j+1}$)")
# plt.plot(time, np.real(w_solI[0, :]))   # Integral of motion
# plt.plot(time, np.angle(w_solI[1, :]))   # They lie (up to precision of the integrator), on the unit circle
# plt.plot(time, np.angle(w_solI[2, :]))   # They lie (up to precision of the integrator), on the unit circle
# plt.plot(time, np.abs(w_solI[3, :]))    # The phase is null
plt.plot(time, y_sol % (2*np.pi))
plt.plot(time, np.angle(w_sol[1, :]) % (2*np.pi))
plt.plot(time, np.angle(w_sol[2, :]) % (2*np.pi))
# plt.plot(time, np.abs(w_sol[3, :]))
# plt.plot(time, np.abs(w_sol[3, :]))

plt.legend()
# plt.ylim([-0.1, 2*np.pi + 0.1])
plt.xlabel("Time t")
# plt.xlabel("New independent variable $y$")
plt.show()