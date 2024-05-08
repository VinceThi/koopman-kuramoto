# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi

plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 10, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 1
coupling = 1/N
theta0 = np.array([0, 2, 4, 6])  # np.random.random(N)

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))


""" Transform the solution theta(t) into another hattheta(t) """


def dilatation(u, a):
    return np.exp(a)*u


# Imaginary, time-independent
# b = 1j*np.ones(len(timelist))                    # Symmetry

# Complex, time-independent
b = (0.5*1j + 2)*np.ones(len(timelist))            # Symmetry

# Real, time-dependent
# b = timelist**2                                  # Trivial symmetry

# Complex, time-independent imaginary part and time-dependent real part
# b = 0.5*1j*np.ones(len(timelist)) + timelist**2    # Symmetry

# Complex, time-dependent imaginary part and time-independent real part
# b = 1j*(1 + timelist**2) + 0.5                   # Not a symmetry

hattheta = []
for i in range(len(timelist)):
    hattheta.append(np.angle(dilatation(np.exp(1j*theta[i, :]), b[i])))
hattheta = np.array(hattheta)

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
plt.figure(figsize=(10, 6))
plt.subplot(211)
theta = theta % (2*np.pi)
hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
        plt.plot(timelist, hattheta[:, i], color=deep[1],
                 linestyle="--", label="Transformed solution $\\hat{\\theta}(t)$")
    else:
        plt.plot(timelist, theta[:, i], color=deep[0])
        plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
plt.ylabel("Phase")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True)

plt.subplot(212)
lw = 3
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(time_derivative_theta[:, i], color=deep[0], linewidth=lw, label="Derivative $\\theta$")
        plt.plot(vector_field_theta[:, i], color=deep[9], linewidth=lw, linestyle="--", label="$F(\\theta)$")
        plt.plot(time_derivative_hattheta[:, i], color=deep[4], label="Derivative $\\hat{\\theta}$")
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--", label="$F(\\hat{\\theta}$)")
    else:
        plt.plot(time_derivative_theta[:, i], linewidth=lw, color=deep[0])
        plt.plot(vector_field_theta[:, i], linewidth=lw, color=deep[9], linestyle="--")
        plt.plot(time_derivative_hattheta[:, i], color=deep[4])
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
plt.ylim([-0.5, 3])
plt.xlabel("Timepoints")
plt.legend()

plt.show()
