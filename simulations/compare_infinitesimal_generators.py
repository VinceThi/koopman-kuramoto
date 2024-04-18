# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz import ws_equations_kuramoto, ws_transformation
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions

plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 20, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 1
coupling = 1/N
theta0 = np.array([0, 2, 4, 6])

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate Watanabe-Strogatz equations under other conditions """

# Arbitrary Z0, phi0, w
Z0, phi0, w = 0.5, 1, np.exp(1j*np.array([0, 1, 2, 3]))
# Z0, phi0, w = 0, 0, np.exp(1j*theta0)

args_ws = (w, coupling, omega)
solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
Z, phi = solution[:, 0], solution[:, 1]

theta_ws = []
for i in range(len(timelist)):
    theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
theta_ws = np.array(theta_ws)

""" Transform the solution of the Kuramoto model with Mobius transformation from unrelated WS equations"""
theta_transformed = []
vector_field = []
vf = []
for i in range(len(timelist)):
    angle = np.angle(ws_transformation(Z[i], phi[i], np.exp(1j*theta[i, :])))
    theta_transformed.append(angle)
    vector_field.append(kuramoto_sakaguchi(0, angle, W, coupling, omega, alpha))
    vf.append(kuramoto_sakaguchi(0, theta[i, :], W, coupling, omega, alpha))
theta_transformed = np.array(theta_transformed)


""" Compute the derivative """
derivative_theta_transformed = np.diff(theta_transformed, axis=0)/dt
vector_field = np.array(vector_field)

d = np.diff(theta, axis=0)/dt
vf = np.array(vf)


plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(d, color=deep[2], label="Derivative $\\theta$")
plt.plot(vf, color=deep[4], linestyle="dotted", label="$F(\\theta)$")
plt.plot(derivative_theta_transformed, color=deep[0], label="Derivative $\\theta$ transformed")
plt.plot(vector_field, color=deep[1], linestyle="--", label="$F$(transformed $\\theta$)")
plt.xlabel("Timepoints")
# plt.legend()

plt.subplot(212)
theta = theta % (2*np.pi)
theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)
theta_transformed = np.where(theta_transformed < 0, 2*np.pi + theta_transformed, theta_transformed)
plt.plot(timelist, theta, color=deep[0], label="Kuramoto")
plt.plot(timelist, theta_ws, color=deep[1], linestyle="--", label="Unrelated WS")
plt.plot(timelist, theta_transformed, color=deep[2], linestyle="dotted", label="Transformed solution")
plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
plt.xlabel("Time $t$")
# plt.legend()
plt.show()
