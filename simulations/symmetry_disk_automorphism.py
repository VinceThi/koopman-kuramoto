# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous, integrate_rk4_non_autonomous
from scipy.integrate import solve_ivp
from dynamics.constants_of_motion import log_cross_ratio_theta
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.symmetries import (determining_equations_real_disk_automorphism, #determining_equations_disk_automorphism,
                                 determining_equations_real_disk_automorphism_kuramoto, nu_function, nu_derivative,
                                 determining_equations_disk_automorphism_bounded)


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
# np.random.seed(2333)
theta0 = 2*np.pi*np.random.random(N)  # np.array([0, 2, 4, 6])  #
print(theta0)

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate determining equations """
args_determining = (omega, coupling)

R0 = 0.1
Phi0 = 2*np.pi/3
Y0 = 0.3
assert R0**2 - Y0**2 + 1 >= 0
print(R0, Phi0, Y0)

solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_real_disk_automorphism,
                                                     np.array([R0, Phi0, Y0]), theta, *args_determining))
# solution = np.array(integrate_rk4_non_autonomous(t0, t1, dt, determining_equations_real_disk_automorphism,
#                                                  np.array([R0, Phi0, Y0]), theta, *args_determining))

R, Phi, Y = solution[:, 0], solution[:, 1], solution[:, 2]
X = np.sqrt(R**2 - Y**2 + 1)
U = X + 1j*Y
V = R*np.exp(1j*Phi)

phi1 = 2*np.arcsin(Y/np.sqrt(1 + R**2))  # np.angle(-U/np.abs(U))
Z1 = R/np.sqrt(1 + R**2)*np.exp(1j*(Phi + phi1/2))  # V/U


""" Integrate bounded determining equations """

# First
rho0 = R0/np.sqrt(1 + R0**2)
phi0 = 2*np.arcsin(Y0/np.sqrt(1 + R0**2))
Psi0 = Phi0 + phi0/2
solution_b = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism_bounded,
                                                       np.array([rho0, Psi0, phi0]), theta, *args_determining))
rho, Psi, phi = solution_b[:, 0], solution_b[:, 1], solution_b[:, 2]

Z = rho*np.exp(1j*Psi)


# Second
# Z0 = rho0*np.exp(1j*Psi0)
# solution_b2 = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism,
#                                                         np.array([Z0, phi0]), theta, *args_determining))
#
# Z2, phi2 = solution_b2[:, 0], solution_b2[:, 1]


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


def disk_automorphism_bounded(Z, phi, z): return (np.exp(1j*phi)*z + Z)/(np.exp(1j*phi)*np.conjugate(Z)*z + 1)


hattheta = []
for i in range(len(timelist)):
    hattheta.append(np.angle(disk_automorphism(U[i], V[i], np.exp(1j*theta[i, :]))))
hattheta = np.array(hattheta)


hattheta_b = []
for i in range(len(timelist)):
    hattheta_b.append(np.angle(disk_automorphism_bounded(Z[i], phi[i], np.exp(1j*theta[i, :]))))
hattheta_b = np.array(hattheta_b)


# hattheta_b2 = []
# for i in range(len(timelist)):
#     hattheta_b2.append(np.angle(disk_automorphism_bounded(Z2[i], phi2[i], np.exp(1j*theta[i, :]))))
# hattheta_b2 = np.array(hattheta_b2)


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

hattheta0_b = hattheta_b[0, :]
hattheta_expected_b = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0_b, *args_dynamics))

# hattheta0_b2 = hattheta_b2[0, :]
# hattheta_expected_b2 = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0_b2, *args_dynamics))

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
plt.figure(figsize=(15, 10))

plt.subplot(231)
theta = theta % (2*np.pi)
# theta1 = theta1 % (2*np.pi)
# theta2 = theta2 % (2*np.pi)
plt.plot(timelist, theta)
plt.ylabel("Solution $\\theta(t)$")
plt.xlabel("Time $t$")

plt.subplot(232)
hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
hattheta_b = np.where(hattheta_b < 0, 2*np.pi + hattheta_b, hattheta_b)
# hattheta_b2 = np.where(hattheta_b2 < 0, 2*np.pi + hattheta_b2, hattheta_b2)
# hattheta1 = np.where(hattheta1 < 0, 2*np.pi + hattheta1, hattheta1)
# hattheta2 = np.where(hattheta2 < 0, 2*np.pi + hattheta2, hattheta2)
hattheta_expected = hattheta_expected % (2*np.pi)
hattheta_expected_b = hattheta_expected_b % (2*np.pi)
# hattheta_expected_b2 = hattheta_expected_b2 % (2*np.pi)
for i in range(len(theta[0, :])):
    if i == 0:
        # plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
        # plt.plot(timelist, theta1[:, i], color=deep[3], label="Solution $\\theta(t)$ ***")
        plt.plot(timelist, hattheta[:, i], color=deep[1],
                 linestyle="--", label="Transformed $\\hat{\\theta}(t)$")
        plt.plot(timelist, hattheta_b[:, i], color=deep[3],
                 linestyle="dotted", label="Transformed $\\hat{\\theta}(t)$ (bounded)")
        # plt.plot(timelist, hattheta_b2[:, i], color=deep[5],
        #          linestyle="dotted", label="Transformed $\\bar{\\theta}(t)$")
        # plt.plot(timelist, hattheta1[:, i], color=deep[3],
        #          linestyle="dotted", label="Transformed solution $\\hat{\\theta}(t)$ *")
        # plt.plot(timelist_bdf, hattheta2[:, i], color=deep[4],
        #          linestyle.="dashdot", label="Transformed solution $\\hat{\\theta}(t)$ **")
        plt.plot(timelist, hattheta_expected[:, i], color=deep[2],
                 linestyle="--", label="Expected $\\hat{\\theta}(t)$")
        # plt.plot(timelist, hattheta_expected_b[:, i], color=deep[4],
        #          linestyle="dotted", label="Expected $\\tilde{\\theta}(t)$")
        # plt.plot(timelist, hattheta_expected_b2[:, i], color=deep[6],
        #          linestyle="dotted", label="Expected $\\bar{\\theta}(t)$")
    else:
        # plt.plot(timelist, theta[:, i], color=deep[0])
        # plt.plot(timelist, theta1[:, i], color=deep[3])
        plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
        plt.plot(timelist, hattheta_b[:, i], color=deep[3], linestyle="dotted")
        # plt.plot(timelist, hattheta_b2[:, i], color=deep[5], linestyle="dotted")
        # plt.plot(timelist, hattheta1[:, i], color=deep[3], linestyle="dotted")
        # plt.plot(timelist_bdf, hattheta2[:, i], color=deep[4], linestyle="dashdot")
        plt.plot(timelist, hattheta_expected[:, i], color=deep[2], linestyle="--")
        # plt.plot(timelist, hattheta_expected_b[:, i], color=deep[4], linestyle="dotted")
        # plt.plot(timelist, hattheta_expected_b2[:, i], color=deep[6], linestyle="dotted")

plt.ylabel("Phase")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)

# plt.subplot(234)
plt.subplot(233)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(vector_field_hattheta_expected[:, i], color=dark_grey, label="Expected $F(\\hat{\\theta}(t)$)")
        # plt.plot(time_derivative_theta[:, i], color=deep[0], label="d$\\theta(t)/$d$t$")
        # plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--", label="$F(\\theta(t))$")
        # plt.plot(time_derivative_hattheta[:, i], color=deep[4], label="d$\\hat{\\theta}(t)/$d$t$")
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--", label="$F(\\hat{\\theta}(t))$")
        # plt.plot(time_derivative_hattheta1[:, i], color=deep[2], linewidth=0.8, label="Derivative $\\hat{\\theta}$ *")
        # plt.plot(vector_field_hattheta1[:, i], color=deep[1], linestyle="dotted", label="$F(\\hat{\\theta})$ *")
        # plt.plot(time_derivative_hattheta2[:, i], color=deep[3], linewidth=0.6,label="Derivative $\\hat{\\theta}$ **")
        # plt.plot(vector_field_hattheta2[:, i], color=deep[8], linestyle="dotted", label="$F(\\hat{\\theta})$ **")
        # plt.plot(time_derivative_hattheta_expected[:, i], color=dark_grey, label="Expected d$\\hat{\\theta}(t)/$d$t$")
    else:
        # plt.plot(time_derivative_hattheta_expected[:, i], color=dark_grey)
        plt.plot(vector_field_hattheta_expected[:, i], color=dark_grey)
        # plt.plot(time_derivative_theta[:, i], color=deep[0])
        # plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--")
        # plt.plot(time_derivative_hattheta[:, i], color=deep[4])
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
        # plt.plot(time_derivative_hattheta1[:, i], linewidth=0.5, color=deep[2])
        # plt.plot(vector_field_hattheta1[:, i], color=deep[1], linestyle="dotted")
        # plt.plot(time_derivative_hattheta2[:, i], color=deep[3], linewidth=0.6)
        # plt.plot(vector_field_hattheta2[:, i], color=deep[8], linestyle="dotted")
plt.ylim([-0.5, 1.5])
plt.xlabel("Timepoints")
plt.legend(loc=1, frameon=True, fontsize=7)

# plt.subplot(232)
plt.subplot(234)
# plt.plot(np.real(U), np.imag(U), label="$U$")
# plt.plot(np.real(V), np.imag(V), label="$V$")
plt.plot(np.real(Z1), np.imag(Z1), label="$Z$ from $U,V$", color=deep[0])
plt.plot(np.real(Z), np.imag(Z), label="$Z$", color=deep[1], linestyle="--")
# plt.plot(np.real(Z2), np.imag(Z2), label="$Z$ *", color=deep[2], linestyle="dotted")
plt.legend(loc=1, frameon=True, fontsize=7)
plt.ylabel("Im")
plt.xlabel("Re")

# plt.subplot(235)
# plt.plot(timelist, log_cross_ratio_theta(theta0[0], theta0[1], theta0[2], theta0[3])*np.ones(len(timelist)),
#          label="From initial conditions", zorder=2)
# plt.plot(timelist, log_cross_ratio_theta(theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]),
#          label="From complete dynamics", zorder=1)
# plt.plot(timelist, log_cross_ratio_theta(hattheta[:, 0], hattheta[:, 1], hattheta[:, 2], hattheta[:, 3]),
#          label="From determining equations", zorder=0)
# # plt.plot(timelist, Phi, label="$\\Phi$")
# # plt.plot(timelist, phi1, label="$\\phi$ from $U,V$")
# plt.ylabel("Log cross ratio $\\ln(c_{1234})$")
# plt.xlabel("Time $t$")
# plt.legend(loc=1, frameon=True, fontsize=7)

plt.subplot(235)  
plt.plot(timelist, X, label="$X$")
# plt.plot(timelist, Y, label="$Y$")
# plt.plot(timelist, R, label="$R$")
plt.plot(timelist, R**2 - Y**2 + 1, label="$R^2 - Y^2 + 1$")
# plt.ylabel("$X$")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)  


p0 = N*coupling/2
p1 = coupling/2*np.sum(np.exp(1j*theta))
p2 = coupling/2*np.sum(np.exp(2*1j*theta))
rho1, phi1 = np.abs(p1), np.angle(p1)
rho2, phi2 = np.abs(p2), np.angle(p2)
chi1 = 2*rho1*np.sin(Phi - phi1)
chi2 = p0 - rho2*np.cos(2*Phi - phi2)
X = np.sqrt(R**2 - Y**2 + 1)
# mu = ((1 - nu_function(X)*np.sqrt(R**2 - Y**2 + 1))/(R**2 - Y**2))*(chi1*Y*R + chi2*R**2)
nu_list = []
nup_list = []
for i in X:
    nu_list.append(nu_function(i))
    nup_list.append(nu_derivative(i))
plt.subplot(236)
# plt.plot(timelist, np.real(mu), label="Re $\\dot{\\nu}/\\nu$")
# plt.plot(timelist, np.imag(mu), label="Im $\\dot{\\nu}/\\nu$")
plt.plot(timelist, nu_list, label="$\\nu(t) = f(X(t))$")
plt.plot(timelist, nup_list, label="$f'(X(t))$")
# plt.ylabel("Im")
# plt.xlabel("Re")
# plt.ylabel("$\\dot{\\nu}/\\nu$")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)
# plt.subplot(235)
# plt.plot(timelist, np.abs(U), label="$|U|$")
# plt.plot(timelist, np.abs(V), label="$|V|$")
# plt.plot(timelist, np.abs(U)**2 - np.abs(V)**2, label="$|U|^2 - |V|^2$")
# plt.xlabel("Time $t$")
# plt.legend()
#
# plt.subplot(224)
# plt.plot(timelist, R/np.sqrt(1 + R**2), label="$\\rho$ from $U,V$", color=deep[0])
# plt.plot(timelist, rho, label="$\\rho$", color=deep[1], linestyle="--")
# # plt.plot(timelist, Phi, label="$\\Phi$")
# # plt.plot(timelist, phi1, label="$\\phi$ from $U,V$")
# # plt.ylabel("$\\rho$")
# plt.xlabel("Time $t$")
# plt.legend(loc=1, frameon=True, fontsize=7)
#
# plt.subplot(236)
# plt.plot(timelist, X, label="$X$")
# plt.plot(timelist, Y, label="$Y$")
# plt.xlabel("Time $t$")
# plt.legend()

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