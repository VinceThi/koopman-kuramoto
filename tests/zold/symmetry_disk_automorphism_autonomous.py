# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.symmetries import disk_automorphism, determining_equations_disk_automorphism


plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 22, 0.005
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 0
coupling = 0.5/N
print(f"omega = {omega} ,", f"coupling = {coupling}")
# np.random.seed(2333)
theta0 = 2*np.pi*np.random.random(N)   # np.array([0, 2, 4, 6])  #
print("theta0 = ", theta0)

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate determining equations """
R0 = 0.5
Phi0 = 0.1
Y0 = 0.2
print("R0, Phi0, Y0 = ", R0, Phi0, Y0)
assert R0**2 - Y0**2 + 1 >= 0
X0 = np.sqrt(R0**2 - Y0**2 + 1)
U0 = X0 + 1j*Y0
V0 = R0*np.exp(1j*Phi0)
hatz0 = disk_automorphism(U0, V0, np.exp(1j*theta0))
hattheta0 = np.angle(hatz0)
hattheta_expected = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0, *args_dynamics))

# args_determining = (W, omega, coupling)
# state0 = np.concatenate([hattheta0, np.array([R0, Phi0, Y0])])
# print(state0)
# solution = np.array(integrate_dopri45(t0, t1, dt, determining_equations_disk_automorphism, state0, *args_determining))
#
args_determining = (omega, coupling)
solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism,
                                                     np.array([R0, Phi0, Y0]), hattheta_expected, *args_determining))
#                                                                        #  theta

R, Phi, Y = solution[:, 0], solution[:, 1], solution[:, 2]
X = np.sqrt(R**2 - Y**2 + 1)
U = X + 1j*Y
V = R*np.exp(1j*Phi)

phi1 = 2*np.arcsin(Y/np.sqrt(1 + R**2))  # np.angle(-U/np.abs(U))
Z1 = R/np.sqrt(1 + R**2)*np.exp(1j*(Phi + phi1/2))  # V/U


""" Integrate bounded determining equations """

# First
# rho0 = R0/np.sqrt(1 + R0**2)
# phi0 = 2*np.arcsin(Y0/np.sqrt(1 + R0**2))
# Psi0 = Phi0 + phi0/2
# solution_b = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism_bounded,
#                                                        np.array([rho0, Psi0, phi0]), theta, *args_determining))
# rho, Psi, phi = solution_b[:, 0], solution_b[:, 1], solution_b[:, 2]
#
# Z = rho*np.exp(1j*Psi)


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


def disk_automorphism_bounded(Z, phi, z): return (np.exp(1j*phi)*z + Z)/(np.exp(1j*phi)*np.conjugate(Z)*z + 1)


def disk_automorphism_real(X, Y, R, Phi, theta):
    # denom = 2*R**2 + 1 + 2*X*R*np.cos(theta - Phi) - 2*Y*R*np.sin(theta - Phi)
    real_num = (X**2 - Y**2)*np.cos(theta) \
        - 2*X*Y*np.sin(theta) + 2*X*R*np.cos(Phi) - 2*Y*R*np.sin(Phi) + R**2*np.cos(theta - 2*Phi)
    imag_num = (X**2 - Y**2)*np.sin(theta) \
        + 2*X*Y*np.cos(theta) + 2*X*R*np.sin(Phi) + 2*Y*R*np.cos(Phi) - R**2*np.sin(theta - 2*Phi)
    return np.arctan2(imag_num, real_num)


hattheta = []
for i in range(len(timelist)):
    hattheta.append(np.angle(disk_automorphism(U[i], V[i], np.exp(1j*theta[i, :]))))
hattheta = np.array(hattheta)


# hattheta_b = []
# for i in range(len(timelist)):
#     hattheta_b.append(np.angle(disk_automorphism_bounded(Z[i], phi[i], np.exp(1j*theta[i, :]))))
# hattheta_b = np.array(hattheta_b)
# 
# 
# hattheta_r = []
# for i in range(len(timelist)):
#     hattheta_r.append(disk_automorphism_real(X[i], Y[i], R[i], Phi[i], theta[i, :]))
# hattheta_r = np.array(hattheta_r)


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


# """ For the initial conditions of the transformed system, what is the expected dynamics ? """
# hattheta0 = hattheta[0, :]
# hattheta_expected = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0, *args_dynamics))

# hattheta0_b = hattheta_b[0, :]
# hattheta_expected_b = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0_b, *args_dynamics))
#
# hattheta0_r = hattheta_r[0, :]
# hattheta_expected_r = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0_r, *args_dynamics))


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
# hattheta_b = np.where(hattheta_b < 0, 2*np.pi + hattheta_b, hattheta_b)
# hattheta_r = np.where(hattheta_r < 0, 2*np.pi + hattheta_r, hattheta_r)
# hattheta_b2 = np.where(hattheta_b2 < 0, 2*np.pi + hattheta_b2, hattheta_b2)
# hattheta1 = np.where(hattheta1 < 0, 2*np.pi + hattheta1, hattheta1)
# hattheta2 = np.where(hattheta2 < 0, 2*np.pi + hattheta2, hattheta2)
hattheta_expected = hattheta_expected % (2*np.pi)
# hattheta_expected_b = hattheta_expected_b % (2*np.pi)
# hattheta_expected_r = hattheta_expected_r % (2*np.pi)
# hattheta_expected_b2 = hattheta_expected_b2 % (2*np.pi)
for i in range(len(theta[0, :])):
    if i == 0:
        # plt.plot(timelist, theta[:, i], color=deep[0], label="Solution $\\theta(t)$")
        # plt.plot(timelist, theta1[:, i], color=deep[3], label="Solution $\\theta(t)$ ***")
        plt.plot(timelist, hattheta[:, i], color=deep[1],
                 linestyle="--", label="Transformed $\\hat{\\theta}(t)$")
        # plt.plot(timelist, hattheta_b[:, i], color=deep[2],
        #          linestyle="dotted", label="Transformed $\\hat{\\theta}(t)$ (bounded)")
        # plt.plot(timelist, hattheta_r[:, i], color=deep[4],
        #          linestyle="dashdot", label="Transformed $\\hat{\\theta}(t)$ (real)")
        # plt.plot(timelist, hattheta_b2[:, i], color=deep[5],
        #          linestyle="dotted", label="Transformed $\\bar{\\theta}(t)$")
        # plt.plot(timelist, hattheta1[:, i], color=deep[3],
        #          linestyle="dotted", label="Transformed solution $\\hat{\\theta}(t)$ *")
        # plt.plot(timelist_bdf, hattheta2[:, i], color=deep[4],
        #          linestyle.="dashdot", label="Transformed solution $\\hat{\\theta}(t)$ **")
        plt.plot(timelist, hattheta_expected[:, i], color=total_color,
                 linestyle="-", label="Expected $\\hat{\\theta}(t)$")
        # plt.plot(timelist, hattheta_expected_r[:, i], color=deep[9],
        #          linestyle="dashdot", label="Expected $\\hat{\\theta}(t)$ (real)")
        # plt.plot(timelist, hattheta_expected_b[:, i], color=deep[4],
        #          linestyle="dotted", label="Expected $\\tilde{\\theta}(t)$")
        # plt.plot(timelist, hattheta_expected_b2[:, i], color=deep[6],
        #          linestyle="dotted", label="Expected $\\bar{\\theta}(t)$")
    else:
        # plt.plot(timelist, theta[:, i], color=deep[0])
        # plt.plot(timelist, theta1[:, i], color=deep[3])
        plt.plot(timelist, hattheta[:, i], color=deep[1], linestyle="--")
        # plt.plot(timelist, hattheta_b[:, i], color=deep[2], linestyle="dotted")
        # plt.plot(timelist, hattheta_r[:, i], color=deep[4], linestyle="dashdot")
        # plt.plot(timelist, hattheta_b2[:, i], color=deep[5], linestyle="dotted")
        # plt.plot(timelist, hattheta1[:, i], color=deep[3], linestyle="dotted")
        # plt.plot(timelist_bdf, hattheta2[:, i], color=deep[4], linestyle="dashdot")
        plt.plot(timelist, hattheta_expected[:, i], color=total_color, linestyle="-")
        # plt.plot(timelist, hattheta_expected_r[:, i], color=deep[9], linestyle="dashdot")
        # plt.plot(timelist, hattheta_expected_b[:, i], color=deep[4], linestyle="dotted")
        # plt.plot(timelist, hattheta_expected_b2[:, i], color=deep[6], linestyle="dotted")
plt.ylabel("Phase")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)

plt.subplot(233)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(vector_field_hattheta_expected[:, i], color=dark_grey, label="Expected $F(\\hat{\\theta}(t)$)")
        # plt.plot(time_derivative_theta[:, i], color=deep[0], label="d$\\theta(t)/$d$t$")
        # plt.plot(vector_field_theta[:, i], color=deep[9], linestyle="--", label="$F(\\theta(t))$")
        plt.plot(time_derivative_hattheta[:, i], color=deep[4], label="d$\\hat{\\theta}(t)/$d$t$")
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
        plt.plot(time_derivative_hattheta[:, i], color=deep[4])
        plt.plot(vector_field_hattheta[:, i], color=deep[6], linestyle="--")
        # plt.plot(time_derivative_hattheta1[:, i], linewidth=0.5, color=deep[2])
        # plt.plot(vector_field_hattheta1[:, i], color=deep[1], linestyle="dotted")
        # plt.plot(time_derivative_hattheta2[:, i], color=deep[3], linewidth=0.6)
        # plt.plot(vector_field_hattheta2[:, i], color=deep[8], linestyle="dotted")
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

plt.subplot(235)
plt.plot(np.real(U), np.imag(U), label="$U$")
plt.plot(np.real(V), np.imag(V), label="$V$")
plt.plot(np.real(Z1), np.imag(Z1), label="$Z$ from $U,V$", color=deep[0])
# plt.plot(np.real(Z), np.imag(Z), label="$Z$", color=deep[1], linestyle="--")
# plt.plot(np.real(Z2), np.imag(Z2), label="$Z$ *", color=deep[2], linestyle="dotted")
plt.legend(loc=1, frameon=True, fontsize=7)
plt.ylabel("Im")
plt.xlabel("Re")


# p0 = N*coupling/2
# p1 = coupling/2*np.sum(np.exp(1j*theta), axis=1)
# p2 = coupling/2*np.sum(np.exp(2*1j*theta), axis=1)
# rho1, phi1 = np.abs(p1), np.angle(p1)
# rho2, phi2 = np.abs(p2), np.angle(p2)
# chi1 = 2*rho1*np.sin(Phi - phi1)
# chi2 = p0 - rho2*np.cos(2*Phi - phi2)
# X = np.sqrt(R**2 - Y**2 + 1)
# mu_list = []
# nu_list = []
# nup_list = []
# for i in range(len(X)):
#     mu_list.append(nu_derivative(X[i])*(chi1[i]*Y[i]*R[i] + chi2[i]*R[i]**2))
#     nu_list.append(nu_function(X[i]))
#     nup_list.append(nu_derivative(X[i]))
# nu_array = np.array(nu_list)
# mu = np.array(mu_list)
plt.subplot(236)
plt.plot(timelist, Y, label="$Y$")
plt.plot(timelist, R, label="$R$")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)

p0 = N*coupling/2
p1 = coupling/2*np.sum(np.exp(1j*theta), axis=1)
p2 = coupling/2*np.sum(np.exp(2*1j*theta), axis=1)
hatp1 = coupling/2*np.sum(np.exp(1j*hattheta), axis=1)
hatp1_expected = coupling/2*np.sum(np.exp(1j*hattheta_expected), axis=1)
plt.subplot(234)
# plt.plot(np.sin(np.linspace(0, 2*np.pi, 1000)), np.cos(np.linspace(0, 2*np.pi, 1000)),
#          linewidth=0.5, linestyle="--", color=total_color)
# plt.plot(np.real(Z), np.imag(Z), label="$Z$", color=deep[1], linestyle="--")
plt.plot(N*coupling/2*np.sin(np.linspace(0, 2*np.pi, 1000)), N*coupling/2*np.cos(np.linspace(0, 2*np.pi, 1000)),
         linewidth=0.5, linestyle="--", color=total_color)
plt.plot(np.real(p1), np.imag(p1), label="$p_1$")
plt.plot(np.real(hatp1), np.imag(hatp1), label="$\\hat{p}_1$")
plt.plot(np.real(hatp1_expected), np.imag(hatp1_expected), label="Expected $\\hat{p}_1$")
plt.legend(loc=1, frameon=True, fontsize=7)
plt.ylabel("Im")
plt.xlabel("Re")



# plt.subplot(232)
# plt.subplot(234)
# plt.plot(np.real(nu_array*V), np.imag(nu_array*V), label="$\\nu V$")
# plt.plot(np.real(nu_array*U), np.imag(nu_array*U), label="$\\nu U$")
# plt.plot(np.real(nu_array*np.conjugate(U)), np.imag(nu_array*np.conjugate(U)), label="$\\nu \\bar{U}$")
# plt.plot(np.real(nu_array*(U - np.conjugate(U))), np.imag(nu_array*(U - np.conjugate(U))), label="$\\nu (U-\\bar{U})$")
# # plt.plot(np.real(U), np.imag(U), label="$U$")
# # plt.plot(np.real(V), np.imag(V), label="$V$")
# # plt.plot(np.real(Z1), np.imag(Z1), label="$Z$ from $U,V$", color=deep[0])
# # plt.plot(np.real(Z), np.imag(Z), label="$Z$", color=deep[1], linestyle="--")
# # plt.plot(np.real(Z2), np.imag(Z2), label="$Z$ *", color=deep[2], linestyle="dotted")
# plt.legend(loc=1, frameon=True, fontsize=7)
# plt.ylabel("Im")
# plt.xlabel("Re")
#
#
# plt.subplot(235)
# plt.plot(timelist, X, label="$X$")
# plt.plot(timelist, R**2 - Y**2 + 1, label="$R^2 - Y^2 + 1$")
# # plt.plot(timelist, X, label="$X$")
# plt.plot(timelist, Y, label="$Y$")
# plt.plot(timelist, R, label="$R$")
# plt.plot(timelist, nu_array*R*np.cos(Phi), label="$\\nu R \\cos(\\Phi)$")
# plt.plot(timelist, nu_array*R*np.sin(Phi), label="$\\nu R \\sin(\\Phi)$")
# plt.plot(timelist, 2*nu_array*Y, label="$-i\\nu(U - \\bar{U}) = 2\\nu Y$")
# # plt.plot(timelist, R**2 - Y**2 + 1, label="$R^2 - Y^2 + 1$")
# # plt.plot(timelist, R/np.sqrt(X**2 + Y**2), label="$R(t)/\\sqrt{X(t)^2 + Y(t)^2}$")
# # plt.plot(timelist, np.arctan2(imag_num, real_num))
# plt.xlabel("Time $t$")
# plt.legend(loc=4, frameon=True, fontsize=7)

plt.show()
