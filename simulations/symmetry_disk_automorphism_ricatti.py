# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi, ricatti
from dynamics.symmetries import disk_automorphism, determining_equations_disk_automorphism


plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 10, 0.0005
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 1
coupling = 0.5/N
print(f"omega = {omega} ,", f"coupling = {coupling}")
# np.random.seed(2333)
theta0 = 2*np.pi*np.random.random(N)   # np.array([0, 2, 4, 6])  #
print("theta0 = ", theta0)

""" Integrate Kuramoto model and the related Ricatti equation """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

args_ricatti = (omega, coupling)
sol_ricatti = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, ricatti, np.exp(1j*theta0), theta, *args_ricatti))
theta_ricatti = np.angle(sol_ricatti)

""" Integrate determining equations """
R0 = 2*np.random.random()
Phi0 = 2*np.pi*np.random.random()
Y0 = np.sqrt(1 + R0**2) - np.random.random()
# Belles valeurs
# omega = 1 , coupling = 0.5/N = 0.125
# theta0 =  [3.7, 5.3, 2.8, 3.8]
# R0, Phi0, Y0 =  0.65, 0.74, 0.6
print("R0, Phi0, Y0 = ", R0, Phi0, Y0)
assert R0**2 - Y0**2 + 1 >= 0
X0 = np.sqrt(R0**2 - Y0**2 + 1)
U0 = X0 + 1j*Y0
V0 = R0*np.exp(1j*Phi0)
args_determining = (omega, coupling)
solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism,
                                                     np.array([R0, Phi0, Y0]), theta, *args_determining))
R, Phi, Y = solution[:, 0], solution[:, 1], solution[:, 2]
X = np.sqrt(R**2 - Y**2 + 1)
U = X + 1j*Y
V = R*np.exp(1j*Phi)
phi1 = 2*np.arcsin(Y/np.sqrt(1 + R**2))  # np.angle(-U/np.abs(U))
Z1 = R/np.sqrt(1 + R**2)*np.exp(1j*(Phi + phi1/2))  # V/U


""" Transform the solution theta(t) into another hattheta(t) """
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


""" Integrate expected transformed trajectory of the --> Ricatti <-- equations """
hatz0 = disk_automorphism(U0, V0, np.exp(1j*theta0))
transformed_ricatti = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, ricatti, hatz0, theta, *args_ricatti))
hattheta_expected = np.angle(transformed_ricatti)


""" Integrate Kuramoto with transformed initial conditions """
args_dynamics = (W, coupling, omega, alpha)
hattheta_kur = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, np.angle(hatz0), *args_dynamics))


""" Integrate bounded determining equations """
# TODO

""" Compare the trajectories vs. the transformed trajectories """
plt.figure(figsize=(6, 6))

plt.subplot(211)
theta = theta % (2*np.pi)
theta_ricatti = theta_ricatti % (2*np.pi)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, theta[:, i], color=reduced_grey, linewidth=2,
                 label="Kuramoto solution $\\theta_{\\mathrm{Kur}}(t)$")
        plt.plot(timelist, theta_ricatti[:, i], color=complete_grey, linestyle="--",
                 label="Equivalent Ricatti solution $\\theta_{\\mathrm{Ric}}(t)$")
    else:
        plt.plot(timelist, theta[:, i], linewidth=2, color=reduced_grey)
        plt.plot(timelist, theta_ricatti[:, i], color=complete_grey, linestyle="--")
plt.ylabel("Phase trajectory")
plt.xlabel("Time $t$")
plt.ylim([-0.2, 2*np.pi+0.2])
plt.legend(loc=1, frameon=True, fontsize=7)

plt.subplot(212)
hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
hattheta_expected = hattheta_expected % (2*np.pi)
hattheta_kur = hattheta_kur % (2*np.pi)
for i in range(len(theta[0, :])):
    if i == 0:
        plt.plot(timelist, hattheta_expected[:, i], color=deep[2],
                 linestyle="-", label="Expected solution $\\hat{\\theta}_{\\mathrm{Ric}}(t)$")
        plt.plot(timelist, hattheta[:, i], color=complete_grey, linewidth=2,
                 linestyle="--", label="Transformed solution $\\hat{\\theta}_{\\mathrm{Ric}}(t)$")
        plt.plot(timelist, hattheta_kur[:, i], color=reduced_grey, linewidth=2,
                 label="Kuramoto for these init. cond.")
    else:
        plt.plot(timelist, hattheta_expected[:, i], color=deep[2], linestyle="-", linewidth=2)
        plt.plot(timelist, hattheta[:, i], color=complete_grey, linestyle="--")
        plt.plot(timelist, hattheta_kur[:, i], color=reduced_grey, linewidth=2)
plt.ylabel("Phase trajectory")
plt.xlabel("Time $t$")
plt.ylim([-0.2, 2*np.pi+0.2])
plt.legend(loc=1, frameon=True, fontsize=7)

plt.show()
