# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous, integrate_rk4_non_autonomous
from scipy.integrate import solve_ivp
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.symmetries import (determining_equations_real_disk_automorphism,
                                 determining_equations_real_disk_automorphism_kuramoto)

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
# np.random.seed(5)
theta0 = 2*np.pi*np.random.random(N)  # np.array([0, 2, 4, 6])  # 

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate determining equations """
args_determining = (omega, coupling)

R0 = 0.5
Phi0 = np.pi/5
Y0 = 0.1
assert R0 > Y0

solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_real_disk_automorphism,
                                                     np.array([R0, Phi0, Y0]), theta, *args_determining))
# solution = np.array(integrate_rk4_non_autonomous(t0, t1, dt, determining_equations_real_disk_automorphism,
#                                                  np.array([R0, Phi0, Y0]), theta, *args_determining))

R, Phi, Y = solution[:, 0], solution[:, 1], solution[:, 2]
X = np.sqrt(R**2 - Y**2 + 1)
U = X + 1j*Y
V = R*np.exp(1j*Phi)

Z = V/U
phi = np.angle(-U/np.abs(U))


# """ Second way to integrate the determining equations : with autonomous equations by including Kuramoto """
# args_determining_kuramoto = (W, omega, coupling)
# detkur0 = np.concatenate([theta0, np.array([R0, Phi0, Y0])])
# solution1 = np.array(integrate_dopri45(t0, t1, dt, determining_equations_real_disk_automorphism_kuramoto,
#                      detkur0, *args_determining_kuramoto))
# theta1, R1, Phi1, Y1 = solution1[:, :N], solution1[:, -3], solution1[:, -2], solution1[:, -1]
# X1 = np.sqrt(R1**2 - Y1**2 + 1)
# U1 = X1 + 1j*Y1
# V1 = R1*np.exp(1j*Phi1)
#
#
# """ Third way to integrate the determining equations : autonomous, including Kuramoto, with solve_ivp, BDF """
# t_span = [t0, t1]
# integration_method = 'BDF'
# rtol = 1e-8
# atol = 1e-12
# sol = solve_ivp(determining_equations_real_disk_automorphism_kuramoto, t_span, detkur0, integration_method,
#                 args=args_determining_kuramoto, rtol=rtol, atol=atol)   # vectorized=True
# solution2 = sol.y.T
# timelist_bdf = sol.t
# theta2, R2, Phi2, Y2 = solution2[:, :N], solution2[:, -3], solution2[:, -2], solution2[:, -1]
# X2 = np.sqrt(R2**2 - Y2**2 + 1)
# U2 = X2 + 1j*Y2
# V2 = R2*np.exp(1j*Phi2)


""" Transform the solution theta(t) into another hattheta(t) """


def disk_automorphism(U, V, z): return (U*z + V)/(np.conjugate(V)*z + np.conjugate(U))
# def disk_automorphism(U, V, z): return (np.conjugate(U)*z - V)/(-np.conjugate(V)*z + U)   # Inverse of above


hattheta = []
for i in range(len(timelist)):
    hattheta.append(np.angle(disk_automorphism(U[i], V[i], np.exp(1j*theta[i, :]))))
hattheta = np.array(hattheta)


# hattheta1 = []
# for i in range(len(timelist)):  # transform the solution I integrated at the same time --
#     hattheta1.append(np.angle(disk_automorphism(U1[i], V1[i], np.exp(1j*theta1[i, :]))))
# hattheta1 = np.array(hattheta1)
#
#
# hattheta2 = []
# for i in range(len(timelist_bdf)):  # transform the solution I integrated at the same time --
#     hattheta2.append(np.angle(disk_automorphism(U2[i], V2[i], np.exp(1j*theta2[i, :]))))
# hattheta2 = np.array(hattheta2)


""" For the initial conditions of the transformed system, what is the expected dynamics ? """
hattheta0 = hattheta[0, :]
hattheta_expected = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0, *args_dynamics))


""" Compute the vector fields to see if we indeed obtained solutions of the Kuramoto model """
vector_field_theta = []
vector_field_hattheta = []
vector_field_hattheta1 = []
vector_field_hattheta2 = []
vector_field_hattheta_expected = []
for i in range(len(timelist)):
    vector_field_theta.append(kuramoto_sakaguchi(0, theta[i, :], W, coupling, omega, alpha))  # To test
    vector_field_hattheta.append(kuramoto_sakaguchi(0, hattheta[i, :], W, coupling, omega, alpha))
    # vector_field_hattheta1.append(kuramoto_sakaguchi(0, hattheta1[i, :], W, coupling, omega, alpha))
    vector_field_hattheta_expected.append(kuramoto_sakaguchi(0, hattheta_expected[i, :], W, coupling, omega, alpha))
# for j in range(len(timelist_bdf)):
#     vector_field_hattheta2.append(kuramoto_sakaguchi(0, hattheta2[j, :], W, coupling, omega, alpha))
vector_field_theta = np.array(vector_field_theta)
vector_field_hattheta = np.array(vector_field_hattheta)
# vector_field_hattheta1 = np.array(vector_field_hattheta1)
# vector_field_hattheta2 = np.array(vector_field_hattheta2)
vector_field_hattheta_expected = np.array(vector_field_hattheta_expected)


""" Compute the time derivative """
time_derivative_theta = np.diff(theta, axis=0)/dt
time_derivative_hattheta = np.diff(hattheta, axis=0)/dt
# time_derivative_hattheta1 = np.diff(hattheta1, axis=0)/dt
# time_derivative_hattheta2 = np.gradient(hattheta2, axis=0)/np.tile(np.gradient(sol.t).reshape((len(timelist_bdf), 1)),
#                                                                    (1, N))
time_derivative_hattheta_expected = np.diff(hattheta_expected, axis=0)/dt


""" Compare the time derivatives vs. the vector fields and the trajectories vs. the transformed trajectories """
plt.figure(figsize=(14, 10))
plt.subplot(231)
theta = theta % (2*np.pi)
# theta1 = theta1 % (2*np.pi)
# theta2 = theta2 % (2*np.pi)
hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
# hattheta1 = np.where(hattheta1 < 0, 2*np.pi + hattheta1, hattheta1)
# hattheta2 = np.where(hattheta2 < 0, 2*np.pi + hattheta2, hattheta2)
hattheta_expected = hattheta_expected % (2*np.pi)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
        # plt.plot(timelist, theta1[:, i], color=deep[3], label="Solution $\\theta(t)$ ***")
        plt.plot(timelist, hattheta[:, i], color=deep[1],
                 linestyle="--", label="Transformed solution $\\hat{\\theta}(t)$")
        # plt.plot(timelist, hattheta1[:, i], color=deep[3],
        #          linestyle="dotted", label="Transformed solution $\\hat{\\theta}(t)$ *")
        # plt.plot(timelist_bdf, hattheta2[:, i], color=deep[4],
        #          linestyle="dashdot", label="Transformed solution $\\hat{\\theta}(t)$ **")
        plt.plot(timelist, hattheta_expected[:, i], color=deep[2],
                 linestyle="--", label="Expected $\\hat{\\theta}(t)$")
    else:
        plt.plot(timelist, theta[:, i], color=deep[0])
        # plt.plot(timelist, theta1[:, i], color=deep[3])
        plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
        # plt.plot(timelist, hattheta1[:, i], color=deep[3], linestyle="dotted")
        # plt.plot(timelist_bdf, hattheta2[:, i], color=deep[4], linestyle="dashdot")
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
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--", label="$F(\\hat{\\theta})$")
        # plt.plot(time_derivative_hattheta1[:, i], color=deep[2], linewidth=0.8, label="Derivative $\\hat{\\theta}$ *")
        # plt.plot(vector_field_hattheta1[:, i], color=deep[1], linestyle="dotted", label="$F(\\hat{\\theta})$ *")
        # plt.plot(time_derivative_hattheta2[:, i], color=deep[3], linewidth=0.6,label="Derivative $\\hat{\\theta}$ **")
        # plt.plot(vector_field_hattheta2[:, i], color=deep[8], linestyle="dotted", label="$F(\\hat{\\theta})$ **")
        plt.plot(time_derivative_hattheta_expected[:, i], color=dark_grey, label="Derivative $\\hat{\\theta}$ (exp)")
        plt.plot(vector_field_hattheta_expected[:, i], color=deep[7], linestyle="dotted",
                 label="$F(\\hat{\\theta}$) (exp)")
    else:
        plt.plot(time_derivative_theta[:, i], color=deep[0])
        plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--")
        plt.plot(time_derivative_hattheta[:, i], color=deep[4])
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
        # plt.plot(time_derivative_hattheta1[:, i], linewidth=0.5, color=deep[2])
        # plt.plot(vector_field_hattheta1[:, i], color=deep[1], linestyle="dotted")
        # plt.plot(time_derivative_hattheta2[:, i], color=deep[3], linewidth=0.6)
        # plt.plot(vector_field_hattheta2[:, i], color=deep[8], linestyle="dotted")
        plt.plot(time_derivative_hattheta_expected[:, i], color=dark_grey)
        plt.plot(vector_field_hattheta_expected[:, i], color=deep[7], linestyle="dotted")
plt.ylim([-0.2, 2])
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





# # -*- coding: utf-8 -*-
# # @author: Vincent Thibeault
#
# from plots.config_rcparams import *
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from dynamics.dynamics import kuramoto_sakaguchi
# from dynamics.symmetries import determining_equations_real_disk_automorphism
#
# plot_trajectories = True
#
# """ Graph parameters """
# N = 4
# W = np.ones((N, N))
#
# """ Dynamical parameters """
# t0, t1 = 0, 12
# t_span = [t0, t1]
# integration_method = 'BDF'
# rtol = 1e-8
# atol = 1e-12
#
# alpha = 0
# omega = 1
# coupling = 0.5/N
# theta0 = np.array([0, 2, 4, 6])  # 2*np.pi*np.random.random(N)  #
#
# """ Integrate Kuramoto model """
# args_dynamics = (W, coupling, omega, alpha)
# sol = solve_ivp(kuramoto_sakaguchi, t_span, theta0, integration_method, args=args_dynamics,
#                 rtol=rtol, atol=atol, vectorized=True)  # ,  jac=jacobian_complete)
# theta = sol.y
# # theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))
#
# """ Integrate determining equations """
# args_determining = (omega, coupling)
#
# R0 = 0.5
# Phi0 = np.pi/5
# Y0 = 0.1
# assert R0 > Y0
#
# solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_real_disk_automorphism,
#                                                      np.array([R0, Phi0, Y0]), theta, *args_determining))
# R, Phi, Y = solution[:, 0], solution[:, 1], solution[:, 2]
# X = np.sqrt(R**2 - Y**2 + 1)
#
# U = X + 1j*Y
# V = R*np.exp(1j*Phi)
#
# Z = V/U
# phi = np.angle(-U/np.abs(U))
#
# """ Transform the solution theta(t) into another hattheta(t) """
#
#
# def disk_automorphism(U, V, z): return (U*z + V)/(np.conjugate(V)*z + np.conjugate(U))
# # def disk_automorphism(U, V, z): return (np.conjugate(U)*z - V)/(-np.conjugate(V)*z + U)   # Inverse of above
#
#
# hattheta = []
# for i in range(len(timelist)):
#     hattheta.append(np.angle(disk_automorphism(U[i], V[i], np.exp(1j*theta[i, :]))))
# hattheta = np.array(hattheta)
#
#
# """ For the initial conditions of the transformed system, what is the expected dynamics ? """
# hattheta0 = hattheta[0, :]
# hattheta_expected = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0, *args_dynamics))
#
#
# """ Compute the vector fields to see if we indeed obtained solutions of the Kuramoto model """
# vector_field_theta = []
# vector_field_hattheta = []
# for i in range(len(timelist)):
#     vector_field_theta.append(kuramoto_sakaguchi(0, theta[i, :], W, coupling, omega, alpha))  # To test
#     vector_field_hattheta.append(kuramoto_sakaguchi(0, hattheta[i, :], W, coupling, omega, alpha))
# vector_field_theta = np.array(vector_field_theta)
# vector_field_hattheta = np.array(vector_field_hattheta)
#
# """ Compute the time derivative """
# time_derivative_theta = np.diff(theta, axis=0)/dt
# time_derivative_hattheta = np.diff(hattheta, axis=0)/dt
#
#
# """ Compare the time derivatives vs. the vector fields and the trajectories vs. the transformed trajectories """
# plt.figure(figsize=(14, 10))
# plt.subplot(231)
# theta = theta % (2*np.pi)
# hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
# hattheta_expected = hattheta_expected % (2*np.pi)
# for i in range(len(theta[0, :])):
#     if i == 0:
#         plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
#         plt.plot(timelist, hattheta[:, i], color=deep[1],
#                  linestyle="--", label="Transformed solution $\\hat{\\theta}(t)$")
#         plt.plot(timelist, hattheta_expected[:, i], color=deep[2],
#                  linestyle="--", label="Expected $\\hat{\\theta}(t)$")
#     else:
#         plt.plot(timelist, theta[:, i], color=deep[0])
#         plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
#         plt.plot(timelist, hattheta_expected[:, i], color=deep[2], linestyle="--")
# plt.ylabel("Phase")
# plt.xlabel("Time $t$")
# plt.legend(loc=1, frameon=True)
#
# plt.subplot(234)
# for i in range(len(theta[0, :])):
#     if i == 0:
#         plt.plot(time_derivative_theta[:, i], color=deep[0], label="Derivative $\\theta$")
#         plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--", label="$F(\\theta)$")
#         plt.plot(time_derivative_hattheta[:, i], color=deep[4], label="Derivative $\\hat{\\theta}$")
#         plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--", label="$F(\\hat{\\theta}$)")
#     else:
#         plt.plot(time_derivative_theta[:, i], color=deep[0])
#         plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--")
#         plt.plot(time_derivative_hattheta[:, i], color=deep[4])
#         plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
# plt.ylim([-0.5, 3])
# plt.xlabel("Timepoints")
# plt.legend()
#
# plt.subplot(232)
# plt.plot(np.real(U), np.imag(U), label="$U$")
# plt.plot(np.real(V), np.imag(V), label="$V$")
# plt.plot(np.real(Z), np.imag(Z), label="$Z$")
# plt.legend()
#
# plt.subplot(235)
# plt.plot(timelist, np.abs(U), label="$|U|$")
# plt.plot(timelist, np.abs(V), label="$|V|$")
# plt.plot(timelist, np.abs(U)**2 - np.abs(V)**2, label="$|U|^2 - |V|^2$")
# plt.xlabel("Time $t$")
# plt.legend()
#
# plt.subplot(233)
# plt.plot(timelist, R, label="$R$")
# plt.plot(timelist, Phi, label="$\\Phi$")
# plt.plot(timelist, phi, label="$\\phi$")
# plt.xlabel("Time $t$")
# plt.legend()
#
# plt.subplot(236)
# plt.plot(timelist, X, label="$X$")
# plt.plot(timelist, Y, label="$Y$")
# plt.xlabel("Time $t$")
# plt.legend()
#
# plt.show()
#