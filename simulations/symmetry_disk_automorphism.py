# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.symmetries import determining_equations_real_disk_automorphism

plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 12, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 1
coupling = 0.5/N
theta0 = np.array([0, 2, 4, 6])  # np.random.random(N)

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate determining equations """
args_determining = (omega, coupling)

# Take 1
# X = 2
# Y0 = 0.5
# V0 = np.sqrt(X**2 + Y0**2 - 1)*np.exp(1j*0.2)
# solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism,
#                                                      np.array([V0, Y0]), theta, *args_determining))
# V, Y = solution[:, 0], solution[:, 1]

# Take 2
# R0 = 0.2
# Phi0 = np.pi/5
# X0 = 1 + 1e-8
# # assert 1 < X0 < np.sqrt(R0**2 + 1)
# # R0 = np.sqrt(X0**2 - 1) + 0.01
# Y0 = -np.sqrt(1 + R0**2 - X0**2)
# print([R0, Phi0, X0, Y0])
# solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_real_disk_automorphism,
#                                                      np.array([R0, Phi0, X0, Y0]), theta, *args_determining))
# R, Phi, X, Y = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]

# Take 3
R0 = 0.2
Phi0 = np.pi/5
Y0 = 0.01
assert R0 > Y0

solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_real_disk_automorphism,
                                                     np.array([R0, Phi0, Y0]), theta, *args_determining))
R, Phi, Y = solution[:, 0], solution[:, 1], solution[:, 2]
X = np.sqrt(R**2 - Y**2 + 1)

U = X + 1j*Y
V = R*np.exp(1j*Phi)

Z = V/U
phi = np.angle(-U/np.abs(U))

""" Transform the solution theta(t) into another hattheta(t) """


def disk_automorphism(U, V, z): return (U*z + V)/(np.conjugate(V)*z + np.conjugate(U))
# def disk_automorphism(U, V, z): return (np.conjugate(U)*z - V)/(-np.conjugate(V)*z + U)   # Inverse of above


hattheta = []
for i in range(len(timelist)):
    hattheta.append(np.angle(disk_automorphism(U[i], V[i], np.exp(1j*theta[i, :]))))
hattheta = np.array(hattheta)


""" For the initial conditions of the transformed system, what is the expected dynamics ? """
hattheta0 = hattheta[0, :]
hattheta_expected = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0, *args_dynamics))


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
plt.figure(figsize=(14, 10))
plt.subplot(231)
theta = theta % (2*np.pi)
hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
hattheta_expected = hattheta_expected % (2*np.pi)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
        plt.plot(timelist, hattheta[:, i], color=deep[1],
                 linestyle="--", label="Transformed solution $\\hat{\\theta}(t)$")
        plt.plot(timelist, hattheta_expected[:, i], color=deep[2],
                 linestyle="--", label="Expected $\\hat{\\theta}(t)$")
    else:
        plt.plot(timelist, theta[:, i], color=deep[0])
        plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
        plt.plot(timelist, hattheta_expected[:, i], color=deep[2], linestyle="--")
plt.ylabel("Phase")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True)

plt.subplot(234)
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

plt.subplot(232)
plt.plot(np.real(U), np.imag(U), label="$U$")
plt.plot(np.real(V), np.imag(V), label="$V$")
plt.plot(np.real(Z), np.imag(Z), label="$Z$")
plt.legend()

plt.subplot(235)
plt.plot(timelist, np.abs(U), label="$|U|$")
plt.plot(timelist, np.abs(V), label="$|V|$")
plt.plot(timelist, np.abs(U)**2 - np.abs(V)**2, label="$|U|^2 - |V|^2$")
plt.xlabel("Time $t$")
plt.legend()

plt.subplot(233)
plt.plot(timelist, R, label="$R$")
plt.plot(timelist, Phi, label="$\\Phi$")
plt.plot(timelist, phi, label="$\\phi$")
plt.xlabel("Time $t$")
plt.legend()

plt.subplot(236)
plt.plot(timelist, X, label="$X$")
plt.plot(timelist, Y, label="$Y$")
plt.xlabel("Time $t$")
plt.legend()

plt.show()
