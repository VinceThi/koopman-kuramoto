# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
from dynamics.integrate import integrate_dopri45, integrate_dopri45_non_autonomous
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.symmetries import determining_equations_disk_automorphism_bounded, nu_function
from tqdm import tqdm
# import pytest


def disk_automorphism(Z, phi, z):
    return (np.exp(1j*phi)*z + Z)/(np.exp(1j*phi)*np.conjugate(Z)*z + 1)


def phase_difference(phi1, phi2):
    delta_phi = np.abs(phi1 - phi2)
    delta_phi = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
    return delta_phi


plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 10, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
rho0_array = np.linspace(0.01, 0.99, 5)
nb_experiments = 5
average_L1_error_array = np.zeros((len(timelist), len(rho0_array)))
for i, rho0 in enumerate(rho0_array):
    print(rho0)
    average_L1_error = np.zeros(len(timelist))
    for _ in range(nb_experiments):
        omega = np.random.random()
        coupling = np.random.random() / N
        print(omega, coupling)
        args_dynamics = (W, coupling, omega, alpha)
        args_determining = (omega, coupling)
        theta0 = 2*np.pi*np.random.random(N)

        """ Integrate Kuramoto model """
        theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))
        z = np.exp(1j*theta)

        """ Integrate determining equations """
        R0 = rho0/np.sqrt(1 - rho0**2)
        Y0 = R0 - R0*np.random.random()
        Phi0 = 2*np.pi*np.random.random()
        phi0 = 2*np.arcsin(Y0/np.sqrt(1 + R0**2))
        Psi0 = Phi0 + phi0/2
        solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, determining_equations_disk_automorphism_bounded,
                                                             np.array([rho0, Psi0, phi0]), theta, *args_determining))
        rho, Psi, phi = solution[:, 0], solution[:, 1], solution[:, 2]
        Z = rho*np.exp(1j*Psi)

        """ Transform the solution theta(t) into another hattheta(t) """
        hattheta = []
        for j in range(len(timelist)):
            hattheta.append(np.angle(disk_automorphism(Z[j], phi[j], np.exp(1j * theta[j, :]))))
        hattheta = np.array(hattheta)

        """ For the initial conditions of the transformed system, what is the expected dynamics ? """
        hattheta0 = hattheta[0, :]
        hattheta_expected = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, hattheta0, *args_dynamics))
        hatz_expected = np.exp(1j*hattheta_expected)

        """ Put every trajectories between 0 and 2*np.pi """
        theta = theta % (2*np.pi)
        hattheta = np.where(hattheta < 0, 2*np.pi + hattheta, hattheta)
        hattheta_expected = hattheta_expected % (2*np.pi)

        """ Plot trajectories """
        if plot_trajectories:
            plt.figure(figsize=(12, 8))

            plt.subplot(231)
            plt.plot(timelist, theta)
            plt.ylabel("Solution $\\theta(t)$")
            plt.xlabel("Time $t$")

            plt.subplot(232)
            for i in range(len(theta[0, :])):
                if i == 0:
                    plt.plot(timelist, hattheta_expected[:, i], color=dark_grey,
                             label="Expected $\\hat{\\theta}(t)$")                                                   
                else:
                    plt.plot(timelist, hattheta_expected[:, i], color=dark_grey)
            plt.plot(timelist, hattheta, linestyle="--")
            plt.ylabel("Transformed $\\hat{\\theta}(t)$")
            plt.xlabel("Time $t$")

            plt.subplot(233)
            X = np.cos(phi/2) / np.sqrt(1 - rho**2)
            nulist = []
            for i in range(len(timelist)):
                nulist.append(nu_function(X[i]))
            plt.plot(timelist, nulist)
            plt.ylabel("$\\nu$")
            plt.xlabel("Time $t$")

            vartheta = np.linspace(0, 2 * np.pi, 1000)
            x = np.cos(vartheta)
            y = np.sin(vartheta)

            plt.subplot(234)
            plt.plot(x, y, color=total_color, linewidth=1)
            for i in range(len(theta[0, :])):
                mobius_numerator = np.exp(1j*phi)*z[:, i] + Z
                if i == 0:
                    plt.plot(np.real(mobius_numerator), np.imag(mobius_numerator), label="Numerator")
                    plt.scatter(np.real(mobius_numerator[0]), np.imag(mobius_numerator[0]))
                else:
                    plt.plot(np.real(mobius_numerator), np.imag(mobius_numerator))
                    plt.scatter(np.real(mobius_numerator[0]), np.imag(mobius_numerator[0]))
            plt.ylabel("Im")
            plt.xlabel("Re")
            plt.axis("equal")
            plt.axhline(0, color='black',linewidth=0.5)
            plt.axvline(0, color='black',linewidth=0.5)
            plt.legend(loc=1, frameon=True, fontsize=7)

            plt.subplot(235)
            plt.plot(x, y, color=total_color, linewidth=1)
            for i in range(len(theta[0, :])):
                mobius_denominator = np.exp(1j*phi)*np.conjugate(Z)*z[:, i] + 1
                if i == 0:
                    plt.plot(np.real(mobius_denominator), np.imag(mobius_denominator), label="Denominator")
                    plt.scatter(np.real(mobius_denominator[0]), np.imag(mobius_denominator[0]))
                else:
                    plt.plot(np.real(mobius_denominator), np.imag(mobius_denominator))
                    plt.scatter(np.real(mobius_denominator[0]), np.imag(mobius_denominator[0]))
            plt.ylabel("Im")
            plt.xlabel("Re")
            plt.axis("equal")                             
            plt.axhline(0, color='black',linewidth=0.5)   
            plt.axvline(0, color='black',linewidth=0.5)   
            plt.legend(loc=1, frameon=True, fontsize=7)

            plt.subplot(236)
            plt.plot(timelist, X)
            plt.ylabel("$X$")
            plt.xlabel("Time $t$")

            plt.show()

        """ Compare transformed solution with expected transformed solution """
        # average_L1_error += np.sum(np.abs(np.angle(hatz_expected - hatz)), axis=1)/N/nb_initial_conditions
        average_L1_error += np.nansum(phase_difference(hattheta_expected, hattheta), axis=1)/N/nb_experiments
        # TODO WARNING         ^^^

    average_L1_error_array[:, i] = average_L1_error
    # print(average_L1_error_array)


plt.figure(figsize=(6, 4))
for k in range(len(rho0_array)):
    plt.plot(timelist, average_L1_error_array[:, k], label=f"$\\rho_0 =${np.round(rho0_array[k], 2)}")
plt.ylabel("Average L1 error")
plt.xlabel("Time $t$")
plt.legend(loc=1, frameon=True, fontsize=7)
plt.show()

# if __name__ == "__main__":
#     pytest.main()
