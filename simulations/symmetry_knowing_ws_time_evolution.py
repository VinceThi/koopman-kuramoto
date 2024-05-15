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
coupling = 0.5/N
theta0 = np.array([0, 2, 4, 6])  # np.random.random(N)

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

# """ Integrate Watanabe-Strogatz equations related to the Kuramoto model """
# Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N)
# args_ws0 = (w, coupling, omega)
# solution_ws = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws0))
# Z, phi = solution_ws[:, 0], solution_ws[:, 1]
#
# # Get theta_ws
# theta_ws = []
# for i in range(len(timelist)):
#     theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
# theta_ws = np.array(theta_ws)

""" Integrate Watanabe-Strogatz equations to transform the solution theta(t) into another hattheta(t) """

# Transform the initial conditions
X, chi = 0.2*np.exp(1j*0.525), 3.4
hattheta0 = np.angle(ws_transformation(X, chi, np.exp(1j*theta0)))

# Generate solutions of WS
Y0, psi0, hatw = get_watanabe_strogatz_initial_conditions(hattheta0, N)
args_ws = (hatw, coupling, omega)
solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Y0, psi0]), *args_ws))
Y, psi = solution[:, 0], solution[:, 1]

# Get hattheta
hattheta = []
for i in range(len(timelist)):
    hattheta.append(np.angle(ws_transformation(Y[i], psi[i], hatw)))
hattheta = np.array(hattheta)

""" Naively transform trajectories """
# tildetheta = []
# for i in range(len(timelist)):
#     tildetheta.append(np.angle(ws_transformation(X, chi, np.exp(1j*theta[i, :]))))
# tildetheta = np.array(tildetheta)

""" Compute the vector fields to see if we indeed obtained solutions of the Kuramoto model """
vector_field_theta = []
vector_field_hattheta = []
for i in range(len(timelist)):
    vector_field_theta.append(kuramoto_sakaguchi(0, theta[i, :], W, coupling, omega, alpha))  # To test
    vector_field_hattheta.append(kuramoto_sakaguchi(0, hattheta[i, :], W, coupling, omega, alpha))
vector_field_theta = np.array(vector_field_theta)
vector_field_hattheta = np.array(vector_field_hattheta)

""" Compute the time derivative """
time_derivative_theta = np.diff(theta, axis=0)/dt
time_derivative_hattheta = np.diff(hattheta, axis=0)/dt

""" Compare the time derivatives vs. the vector fields and the trajectories vs. the transformed trajectories """
plt.figure(figsize=(9, 6))
plt.subplot(211)
theta = theta % (2*np.pi)
hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
# tildetheta = np.where(tildetheta < 0, 2*np.pi + tildetheta, tildetheta)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
        plt.plot(timelist, hattheta[:, i], color=deep[1],
                 linestyle="--", label="Transformed solution $\\hat{\\theta}(t)$")
        # plt.plot(timelist, tildetheta[:, i], color=deep[2], linestyle="dotted",
        #  label="Solution $\\tilde{\\theta}(t)$")
    else:
        plt.plot(timelist, theta[:, i], color=deep[0])
        plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
        # plt.plot(timelist, tildetheta[:, i], color=deep[2], linestyle="dotted")
# plt.plot(timelist, np.angle(np.sum(np.exp(1j*theta), axis=1)/np.sum(np.exp(-1j*theta), axis=1)))
plt.ylabel("Phase")  # $\\theta_1(t), ..., \\theta_N(t)$")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True)

plt.subplot(212)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(time_derivative_theta[:, i], color=deep[0], label="Derivative $\\theta$")
        plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--", label="$F(\\theta)$")
        plt.plot(time_derivative_hattheta[:, i], color=deep[4], label="Derivative $\\hat{\\theta}$")
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--", label="$F(\\hat{\\theta}$)")
    else:
        plt.plot(time_derivative_theta[:, i], color=deep[0])
        plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--")
        plt.plot(time_derivative_hattheta[:, i], color=deep[4])
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
plt.ylim([-0.5, 3])
plt.xlabel("Timepoints")
plt.legend()

plt.show()
