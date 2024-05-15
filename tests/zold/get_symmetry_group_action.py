# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.symmetries import infinitesimal_condition_symmetry_kuramoto, \
    infinitesimal_condition_symmetry_kuramoto_2, rfun

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
theta0 = np.random.random(N)        # np.array([0, 2, 4, 6])  #

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate determining equations """
p1 = coupling/2*np.sum(np.exp(1j*theta), axis=1)
args_determining = (omega, )
Z0, phi0 = 0.001*np.exp(1j*0.5), 0  # 3*np.pi - 0.1  # 0.8
solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, infinitesimal_condition_symmetry_kuramoto,
                                                     np.array([Z0, np.conjugate(Z0), phi0]), p1, *args_determining))
Z, Zbar, phi = solution[:, 0], solution[:, 1], solution[:, 2]


""" Integrate determining equations (Take 2) """
X0, Y0 = np.real(Z0), np.imag(Z0)
solution2 = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, infinitesimal_condition_symmetry_kuramoto_2,
                                                      np.array([X0, Y0, phi0]), p1, *args_determining))
X, Y, phi2 = solution2[:, 0], solution2[:, 1], solution2[:, 2]
Z2 = X + 1j*Y


""" Transform the solution theta(t) into another hattheta(t) """


def disk_automorphism(Z, Zbar, phi, u): return (np.exp(1j*phi)*u + Z)/(1 + np.exp(1j*phi)*Zbar*u)
# def disk_automorphism(Z, phi, u): return (np.exp(1j*phi)*u + Z)/(1 + np.exp(1j*phi)*np.conjugate(Z)*u)


hattheta = []
for i in range(len(timelist)):
    hattheta.append(np.angle(disk_automorphism(Z[i], Zbar[i], phi[i], np.exp(1j*theta[i, :]))))
    # hattheta.append(np.angle(disk_automorphism(Z[i], phi[i], np.exp(1j*theta[i, :]))))
hattheta = np.array(hattheta)


hattheta2 = []
for i in range(len(timelist)):
    hattheta2.append(np.angle(disk_automorphism(Z2[i], np.conjugate(Z2[i]), phi2[i], np.exp(1j*theta[i, :]))))
    # hattheta.append(np.angle(disk_automorphism(Z[i], phi[i], np.exp(1j*theta[i, :]))))
hattheta2 = np.array(hattheta2)
print(np.shape(hattheta2))

""" Compute the vector fields to see if we indeed obtained solutions of the Kuramoto model """
vector_field_theta = []
vector_field_hattheta = []
vector_field_hattheta2 = []
for i in range(len(timelist)):
    vector_field_theta.append(kuramoto_sakaguchi(0, theta[i, :], W, coupling, omega, alpha))  # To test
    vector_field_hattheta.append(kuramoto_sakaguchi(0, hattheta[i, :], W, coupling, omega, alpha))
    vector_field_hattheta2.append(kuramoto_sakaguchi(0, hattheta2[i, :], W, coupling, omega, alpha))
vector_field_theta = np.array(vector_field_theta)
vector_field_hattheta = np.array(vector_field_hattheta)
vector_field_hattheta2 = np.array(vector_field_hattheta2)

""" Compute the time derivative """
time_derivative_theta = np.diff(theta, axis=0)/dt
time_derivative_hattheta = np.diff(hattheta, axis=0)/dt
time_derivative_hattheta2 = np.diff(hattheta2, axis=0)/dt

""" Plot r """
r = rfun(Z2, np.conjugate(Z2), phi2)
plt.plot(timelist, np.real(r), label="Real(r)")
plt.plot(timelist, np.imag(r), label="Imag(r)")
# plt.plot(timelist, phi, label="$\\varphi$")
plt.plot(timelist, phi2, label="$\\varphi_2$")
plt.legend()
plt.show()

""" Compare the time derivatives vs. the vector fields and the trajectories vs. the transformed trajectories """
plt.figure(figsize=(12, 8))
plt.subplot(221)
theta = theta % (2*np.pi)
hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
hattheta2 = np.where(hattheta2 < 0, 2*np.pi + hattheta2, hattheta2)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
        # plt.plot(timelist, hattheta[:, i], color=deep[1],
        #          linestyle="--", label="Transformed solution $\\hat{\\theta}(t)$")
        plt.plot(timelist, hattheta2[:, i], color=deep[2],
                 linestyle="dotted", label="Transformed solution $\\hat{\\theta}_2(t)$")
    else:
        plt.plot(timelist, theta[:, i], color=deep[0])
        # plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
        plt.plot(timelist, hattheta2[:, i], color=deep[2], linestyle="dotted")
plt.ylabel("Phase")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True)

plt.subplot(222)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(time_derivative_theta[:, i], color=deep[0], label="Derivative $\\theta$")
        plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--", label="$F(\\theta)$")
        # plt.plot(time_derivative_hattheta[:, i], color=deep[4], label="Derivative $\\hat{\\theta}$")
        # plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--", label="$F(\\hat{\\theta}$)")
        plt.plot(time_derivative_hattheta2[:, i], color=deep[1], label="Derivative $\\hat{\\theta}_2$")
        plt.plot(vector_field_hattheta2[:, i], color=deep[3], linestyle="--", label="$F(\\hat{\\theta}_2$)")
    else:
        plt.plot(time_derivative_theta[:, i], color=deep[0])
        plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--")
        # plt.plot(time_derivative_hattheta[:, i], color=deep[4])
        # plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
        plt.plot(time_derivative_hattheta2[:, i], color=deep[1])
        plt.plot(vector_field_hattheta2[:, i], color=deep[3], linestyle="--")
plt.ylim([-0.5, 3])
plt.xlabel("Timepoints")
plt.legend()

plt.subplot(223)
# plt.plot(np.real(Z), np.imag(Z), label="$Z$")
# plt.plot(np.real(Zbar), np.imag(Zbar), label="$\\bar{Z}$")
plt.plot(np.real(X), np.real(Y), label="$Z_2$")
plt.legend()

plt.subplot(224)
phi = np.real(phi) % (2*np.pi)
phi2 = np.real(phi2) % (2*np.pi)
# plt.plot(timelist, phi, label="$\\varphi$")
plt.plot(timelist, phi2, label="$\\varphi_2$")
plt.xlabel("Time $t$")
plt.legend()

plt.show()
