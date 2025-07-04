# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.dynamics import kuramoto_sakaguchi
from scipy.integrate import solve_ivp


N = 5

""" Dynamical parameters """
t0, t1, dt = 0.00001, 5, 0.005
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 1
coupling = 1
W = np.ones((N, N))
theta0 = np.random.uniform(0, 2*np.pi, N)


""" Integrate Kuramoto dynamics """
rtol = 1e-8
atol = 1e-12
args_dynamics = (W, coupling, omega, alpha)
thet = solve_ivp(kuramoto_sakaguchi, [t0, t1], theta0, 'BDF', t_eval=np.linspace(t0, t1, 10000), args=args_dynamics, rtol=rtol, atol=atol)
theta = np.array(thet.y)
timet = np.array(thet.t)

""" Integrate Kuramoto dynamics under the change of variable """


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
z0 = np.exp(1j*theta0)
w10 = ((z0[2] - z0[1])*(z0[3] - z0[0]))/((z0[2] - z0[0])*(z0[3] - z0[1]))
c20 = ((z0[3] - z0[1])*(z0[4] - z0[2]))/((z0[3] - z0[2])*(z0[4] - z0[1]))
w0 = np.array([w10, z0[1], z0[2], w10*t0, t0/c20])
y0, y1 = theta0[0], theta[0, -1]
sol = solve_ivp(kuramoto_dwdy, [y0, y1], w0, 'BDF', t_eval=np.linspace(y0, y1, 10000), args=args_dyn, rtol=rtol, atol=atol)
w_sol = sol.y
y_sol = sol.t

w1 = w_sol[0, :]
w2 = w_sol[1, :]
w3 = w_sol[2, :]
w4 = w_sol[3, :]
w5 = w_sol[4, :]

time = w4/w1
plt.plot(time)
plt.show()
theta1 = y_sol % (2*np.pi)
theta2 = np.angle(w_sol[1, :]) % (2*np.pi)
theta3 = np.angle(w_sol[2, :]) % (2*np.pi)
z4 = ((w3 - np.exp(1j*theta1))*w1*w2 - (w3 - w2)*np.exp(1j*theta1))/((w3 - np.exp(1j*theta1))*w1 - (w3 - w2))
theta4 = np.angle(z4) % (2*np.pi)
z5 = ((z4 - w3)*w2*w4 - (z4 - w2)*w1*w3*w5)/((z4 - w3)*w4 - (z4 - w2)*w1*w5)
theta5 = np.angle(z5) % (2*np.pi)


""" Compare solutions """

plt.figure(figsize=(8, 5))
for i in range(N):
    plt.plot(timet, theta[i, :] % (2*np.pi), label=f"$\\theta_{i+1}$")
for j, sol in enumerate([theta1, theta2, theta3, theta4, theta5]):
    plt.plot(time, sol, color=dark_grey, label=f"$\\hat\\theta_{j+1}$", linestyle="--", linewidth=0.8)
plt.ylim([-0.1, 2*np.pi + 0.1])
plt.xlabel("Time t")
plt.legend()
plt.show()
