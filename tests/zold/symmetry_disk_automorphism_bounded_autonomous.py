# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.symmetries import nu_function, nu_derivative, determining_equations_disk_automorphism_bounded


def disk_automorphism_bounded(Z, phi, z):
    return (np.exp(1j*phi)*z + Z)/(np.exp(1j*phi)*np.conjugate(Z)*z + 1)
# def disk_automorphism_bounded(Z, phi, z):
#     return np.exp(-1j*phi)*(z - Z)/(1 - np.conjugate(Z)*z)


plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 25, 0.005
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 0
coupling = 0.5/N
print(f"omega = {omega}", f"coupling = {coupling}", f"N*coupling/2 = {N*coupling/2}")
theta0 = np.array([0, 1.5, 3, 4.5])  # 2*np.pi*np.random.random(N)  #
print("theta0 = ", theta0)

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate determining equations """
args_determining = (omega, coupling)

# R0 = 0.2
# Phi0 = 5.5  # 2*np.pi*np.random.random()
# Y0 = 0.2
# print("R0, Phi0, Y0 = ", R0, Phi0, Y0)

rho0 = 0.5  # R0/np.sqrt(1 + R0**2)
phi0 = 1  # 2*np.arcsin(Y0/np.sqrt(1 + R0**2))
Psi0 = 1    # Phi0 + phi0/2
print("rho0, phi0, Psi0", rho0, phi0, Psi0)
# theta_initial = np.array([[0], [2], [4], [6]])
# theta_init = theta_initial@np.ones((1, len(timelist)))
# theta_init = theta_init.T

Z0 = rho0*np.exp(1j*Psi0)
hattheta_b0 = np.angle(disk_automorphism_bounded(Z0, phi0, np.exp(1j*theta0)))

""" For the initial conditions of the transformed system, what is the expected dynamics ? """
hattheta_expected_b = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta_b0, *args_dynamics))

# hattheta_b0 = hattheta_b0.reshape((N, 1))
# hattheta_init = hattheta_b0@np.ones((1, len(timelist)))
# hattheta_init = hattheta_init.T


solution_b = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism_bounded,
                                                       np.array([rho0, Psi0, phi0]), hattheta_expected_b,
                                                       *args_determining))          #  hattheta_init, #
rho, Psi, phi = solution_b[:, 0], solution_b[:, 1], solution_b[:, 2]
Z = rho*np.exp(1j*Psi)


""" Transform the solution theta(t) into another hattheta(t) """

hattheta_b = []
for i in range(len(timelist)):
    hattheta_b.append(np.angle(disk_automorphism_bounded(Z[i], phi[i], np.exp(1j*theta[i, :]))))
hattheta_b = np.array(hattheta_b)


""" Compute the vector fields to see if we indeed obtained solutions of the Kuramoto model """
vector_field_theta = []
vector_field_hattheta = []
vector_field_hattheta_expected = []
for i in range(len(timelist)):
    vector_field_theta.append(kuramoto_sakaguchi(0, theta[i, :], W, coupling, omega, alpha))  # To test
    vector_field_hattheta.append(kuramoto_sakaguchi(0, hattheta_b[i, :], W, coupling, omega, alpha))
    vector_field_hattheta_expected.append(kuramoto_sakaguchi(0, hattheta_expected_b[i, :], W, coupling, omega, alpha))
vector_field_theta = np.array(vector_field_theta)
vector_field_hattheta = np.array(vector_field_hattheta)
vector_field_hattheta_expected = np.array(vector_field_hattheta_expected)


""" Compute the time derivative """
time_derivative_theta = np.diff(theta, axis=0)/dt
time_derivative_hattheta = np.diff(hattheta_b, axis=0)/dt
time_derivative_hattheta_expected = np.diff(hattheta_expected_b, axis=0)/dt


""" Compare the time derivatives vs. the vector fields and the trajectories vs. the transformed trajectories """
plt.figure(figsize=(15, 10))

plt.subplot(231)
theta = theta % (2*np.pi)
plt.plot(timelist, theta)
plt.ylim([-0.05, 2*np.pi+0.05])
plt.ylabel("Solution $\\theta(t)$")
plt.xlabel("Time $t$")

plt.subplot(232)
hattheta_b = np.where(hattheta_b < 0, 2*np.pi + hattheta_b, hattheta_b)
hattheta_expected_b = hattheta_expected_b % (2*np.pi)
Psi = Psi % (2*np.pi)
plt.plot(timelist, Psi, label="$\\Psi$", linewidth=3)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, hattheta_b[:, i], color=deep[1],
                 linestyle="--", label="Transformed $\\hat{\\theta}(t)$")
        plt.plot(timelist, hattheta_expected_b[:, i], color=total_color,
                 linestyle="-", label="Expected $\\hat{\\theta}(t)$")
    else:
        plt.plot(timelist, hattheta_b[:, i], color=deep[1], linestyle="--")
        plt.plot(timelist, hattheta_expected_b[:, i], color=total_color, linestyle="-")
plt.ylim([-0.05, 2*np.pi+0.05])
plt.ylabel("Phase")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)

plt.subplot(233)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(vector_field_hattheta_expected[:, i], color=dark_grey, label="Expected $F(\\hat{\\theta}(t)$)")
        # plt.plot(time_derivative_theta[:, i], color=deep[0], label="d$\\theta(t)/$d$t$")
        # plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--", label="$F(\\theta(t))$")
        # plt.plot(time_derivative_hattheta[:, i], color=deep[4], label="d$\\hat{\\theta}(t)/$d$t$")
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--", label="$F(\\hat{\\theta}(t))$")
    else:
        # plt.plot(time_derivative_hattheta_expected[:, i], color=dark_grey)
        plt.plot(vector_field_hattheta_expected[:, i], color=dark_grey)
        # plt.plot(time_derivative_theta[:, i], color=deep[0])
        # plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--")
        # plt.plot(time_derivative_hattheta[:, i], color=deep[4])
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
plt.ylim([-0.5, 1.5])
plt.xlabel("Timepoints")
plt.legend(loc=1, frameon=True, fontsize=7)


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


p0 = N*coupling/2
p1 = coupling/2*np.sum(np.exp(1j*theta), axis=1)
p2 = coupling/2*np.sum(np.exp(2*1j*theta), axis=1)
rho1, phi1 = np.abs(p1), np.angle(p1)
rho2, phi2 = np.abs(p2), np.angle(p2)
chi1 = 2*rho1*np.sin(Psi - phi/2 - phi1)
chi2 = p0 - rho2*np.cos(2*Psi - phi - phi2)
X = np.cos(phi/2)/np.sqrt(1 - rho**2)
mu_list = []
nu_list = []
nup_list = []
for i in range(len(X)):
    mu_list.append(nu_derivative(X[i])*rho[i]/(1 - rho[i]**2)*(np.sin(phi[i]/2)*chi1[i] + rho[i]*chi2[i]))
    nu_list.append(nu_function(X[i]))
    nup_list.append(nu_derivative(X[i]))
nu_array = np.array(nu_list)
# mu = np.array(mu_list)
plt.subplot(236)
# plt.plot(timelist, np.real(mu), label="Re $\\dot{\\nu}/\\nu$")
# plt.plot(timelist, np.imag(mu), label="Im $\\dot{\\nu}/\\nu$")
plt.plot(timelist, nu_list, label="$\\nu(t) = f(X(t))$")
plt.plot(timelist, nup_list, label="$f'(X(t))$")
plt.plot(timelist, mu_list, label="$\\mu(t)$")
# plt.ylabel("Im")
# plt.xlabel("Re")
# plt.ylabel("$\\dot{\\nu}/\\nu$")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)


hatp1 = coupling/2*np.sum(np.exp(1j*hattheta_b), axis=1)
plt.subplot(234)
plt.plot(np.sin(np.linspace(0, 2*np.pi, 1000)), np.cos(np.linspace(0, 2*np.pi, 1000)),
         linewidth=0.5, linestyle="--", color=total_color)
plt.plot(np.real(Z), np.imag(Z), label="$Z$", color=deep[1], linestyle="--")
plt.plot(N*coupling/2*np.sin(np.linspace(0, 2*np.pi, 1000)), N*coupling/2*np.cos(np.linspace(0, 2*np.pi, 1000)),
         linewidth=0.5, linestyle="--", color=total_color)
plt.plot(np.real(p1), np.imag(p1), label="$p_1$")
plt.plot(np.real(hatp1), np.imag(hatp1), label="$\\hat{p}_1$")
plt.legend(loc=1, frameon=True, fontsize=7)
plt.ylabel("Im")
plt.xlabel("Re")


phi = phi % (2*np.pi)
plt.subplot(235)
plt.plot(timelist, rho, label="$\\rho$")
plt.plot(timelist, phi, label="$\\phi$")
plt.xlabel("Time $t$")
plt.legend(loc=4, frameon=True, fontsize=7)


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
