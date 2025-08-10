# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.symmetries import ode_symmetry_action_calS, disk_automorphism_bounded
from dynamics.constants_of_motion import log_cross_ratio_theta, cross_ratio_theta
from tqdm import tqdm
from scipy.integrate import solve_ivp
from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45
import time
import json
from pathlib import Path
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog
# import tkinter.simpledialog
# from tkinter import messagebox


plot_Zphi = False
plot_trajectories = True

""" Partition parameters"""
sizes_monomial = np.array([1], dtype=int)
sizes_crossratio = np.array([4, 50], dtype=int)
size_nonintegrable = np.array([], dtype=int)
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
q = len(sizes)  # Number of parts
N = np.sum(sizes, dtype=int)
m = len(sizes_monomial)
c = len(sizes_crossratio)

""" Get weight matrix """
random_exponents = np.array([1])  # np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
probabilities_monomial = np.array([0])
probabilities_crossratio = np.array([[1, 0, 0],
                                     [1, 0., 0.8]])
probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
weights_monomial = np.array([1])  # np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
meanW = 1
stdW = 1
weights_crossratio = np.random.normal(meanW, stdW, (len(sizes_crossratio), N))
weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                            probabilities=probabilities_dict, weights=weights_dict)
coupling = 1
print("W = \n", np.round(W, 3))

""" Get phase-lag matrix """
probabilities_monomial2 = np.array([0])
probabilities_crossratio2 = np.array([[1, 0, 0],
                                      [1, 0., 0.9]])
probabilities_nonintegrable2 = []
probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
meanalpha = 0
stdalpha = 1
phaselags_monomial = np.random.normal(meanalpha, stdalpha, (sum(sizes_monomial), sum(sizes_monomial))) % (np.pi/2)
phaselags_crossratio = np.random.normal(meanalpha, stdalpha, (len(sizes_crossratio), N))
phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
alpha, chi = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                     probabilities=probabilities_dict2, phaselags=phaselags_dict)
print("alpha = \n", np.round(alpha, 3))

""" calA """
cal_A = calA(coupling, C, chi)
print("calA = \n", np.round(cal_A, 3))


""" Natural frequencies"""
omega = random_gaussian_frequencies_pintegrable(m, c, sizes, cal_A, 0.5, 1)
omega[0] = 0   # omega[1] + 2*np.imag(cal_A[0, 0])
omega[1] = -2*np.imag(cal_A[0, 0])
omega[2] = -2*np.imag(cal_A[0, 0])
omega[3] = -2*np.imag(cal_A[0, 0])
omega[4] = -2*np.imag(cal_A[0, 0])
Omega1 = omega[1] - 2*np.imag(cal_A[0, 1])
Omega2 = omega[5] - 2*np.imag(cal_A[1, 5])
print("omega = ", omega)
print("Omega1 = ", Omega1)
print("Omega2 = ", Omega2)

""" Initial conditions """
N = len(alpha[0])
# np.random.seed(12)
theta0 = np.random.uniform(0, 2*np.pi, N)
theta0[0] = 0


""" Action of the symmetry on the cross-ratio c_1234 """
theta3 = [theta0[3]]
theta4 = [theta0[4]]
theta5 = [theta0[5]]
cross_ratio_vs_epsilon = [cross_ratio_theta(theta0[0], theta0[1], theta0[2], theta0[3])]
epsilon_array = np.linspace(0.001, 2, 50)
for epsilon in epsilon_array:
    """ Integrate symmetry """
    Z0, phi0 = 0, 0
    epsilon0 = 0

    zs_t = np.exp(1j * theta0[0])

    # Peripheral oscillators 3, 4, 5 (indices 2,3,4)
    z1_t = np.exp(1j * theta0[2:1 + sizes_crossratio[0]])
    args_ws1 = (z1_t, cal_A[0, 0], cal_A[0, 2:1 + sizes_crossratio[0]], Omega1 - omega[0], zs_t)
    solution1 = solve_ivp(ode_symmetry_action_calS, [epsilon0, epsilon],
                          np.array([Z0, phi0], dtype=complex), vectorized=True,
                          args=args_ws1, rtol=1e-08, atol=1e-10)
    Z1, phi1 = solution1.y[0, :], solution1.y[1, :]
    if plot_Zphi:
        plt.plot(np.real(Z1), np.imag(Z1))
        plt.show()
    theta_transformed_1 = np.angle(disk_automorphism_bounded(Z1[-1], phi1[-1], z1_t))
    theta3.append(theta_transformed_1[0])
    theta4.append(theta_transformed_1[1])
    theta5.append(theta_transformed_1[2])
    cross_ratio_vs_epsilon.append(cross_ratio_theta(theta0[0], theta0[1],
                                                    theta_transformed_1[0], theta_transformed_1[1]))

theta3 = np.array(theta3)
theta4 = np.array(theta4)
theta5 = np.array(theta5)

theta3 = np.where(theta3 < 0, 2 * np.pi + theta3, theta3)
theta4 = np.where(theta4 < 0, 2 * np.pi + theta4, theta4)
theta5 = np.where(theta5 < 0, 2 * np.pi + theta5, theta5)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
ax1.plot(np.concatenate([[0], epsilon_array]), cross_ratio_vs_epsilon)
ax1.set_ylabel("Cross-ratio $c_{1234}($exp$(\\varepsilon\\mathcal{S})z)$")
ax1.set_xlabel("$\\varepsilon$")

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot(theta02, theta3, theta4)
ax2 = fig.add_subplot(122)
ax2.plot(theta3, theta4)
ax2.set_ylabel("$\\theta_4$")
ax2.set_xlabel("$\\theta_3$")

plt.show()
app = QApplication(sys.argv)
reply = QMessageBox.question(None, "Python", "Would you like to save the parameters, the data, and the plot?",
                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
if reply == QMessageBox.StandardButton.Yes:
    filename, ok = QInputDialog.getText(None, "File:", "Enter your file name")
    if ok:
        print("File name:", filename)
    SCRIPT_DIR = Path(__file__).resolve().parent   # Get current script location
    REPO_ROOT = SCRIPT_DIR.parent      # Go to repo root (adjust this based on how deep your script is)
    path = REPO_ROOT / 'simulations' / 'kooku1_fig3_data'  # Path to the data file
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    parameters_dictionary = {"N": N, "sizes_monomial": sizes_monomial, "sizes_crossratio":sizes_crossratio,
                             "size_nonintegrable":size_nonintegrable, "W": W, "alpha": alpha, "coupling": coupling,
                             "omega": omega.tolist(), "Omega1": Omega1, "Omega2": Omega2, "C": C, "chi": chi,
                             "cal_A": cal_A, "meanW": meanW, "stdW": stdW, "meanalpha": meanalpha, "stdalpha": stdalpha,
                             "theta0": theta0, "random_exponents": random_exponents,
                             "probabilities_monomial": probabilities_monomial,
                             "probabilities_crossratio": probabilities_crossratio,
                             "probabilities_nonintegrable2": probabilities_nonintegrable2,
                             "probabilities_monomial2": probabilities_monomial2,
                             "probabilities_crossratio2": probabilities_crossratio2,
                             }
    print(path)
    fig.savefig(path / f'{timestr}_{filename}_symmetry_change_invariant_sets_kuramoto.pdf')
    fig.savefig(path / f'{timestr}_{filename}_symmetry_change_invariant_sets_kuramoto.png')
    with open(path / f'{timestr}_{filename}_kuramoto_parameters_dictionary.json', 'w') as outfile:
        json.dump(parameters_dictionary, outfile)


if plot_trajectories:
    """ Integrate complete dynamics """
    theta0 = np.random.uniform(0, 2 * np.pi, N)
    theta0[0] = 0
    theta0_source = theta0[0]
    t0, t1, dt = 0, 20, 0.01
    timelist = np.linspace(t0, t1, int(t1 / dt))
    args_dynamics = (W, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
    theta = np.where(theta < 0, 2 * np.pi + theta, theta)

    """ Integrate symmetry """
    Z0, phi0 = 0, 0
    epsilon0 = 0
    epsilon1 = 0.4
    epsilon2 = 0.2
    theta_transformed_1 = []
    theta_transformed_2 = []
    for t in tqdm(range(len(timelist))):
        zs_t = np.exp(1j * theta[t, 0])

        # First peripheral oscillators
        z1_t = np.exp(1j * theta[t, 2:1 + sizes_crossratio[0]])
        args_ws1 = (z1_t, cal_A[0, 0], cal_A[0, 2:1 + sizes_crossratio[0]], Omega1 - omega[0], zs_t)
        solution1 = solve_ivp(ode_symmetry_action_calS, [epsilon0, epsilon1],
                              np.array([Z0, phi0], dtype=complex), vectorized=True,
                              args=args_ws1, rtol=1e-08, atol=1e-10)
        Z1, phi1 = solution1.y[0, :], solution1.y[1, :]
        if plot_Zphi:
            plt.plot(np.real(Z1), np.imag(Z1))
            plt.show()
        theta_transformed_1.append(np.angle(disk_automorphism_bounded(Z1[-1], phi1[-1], z1_t)))

        # Second peripheral oscillators
        z2_t = np.exp(1j * theta[t, 1 + sizes_crossratio[0]:])
        args_ws2 = (z2_t, cal_A[1, 0], cal_A[1, 1 + sizes_crossratio[0]:], Omega2 - omega[0], zs_t)
        solution2 = solve_ivp(ode_symmetry_action_calS, [epsilon0, epsilon2],
                              np.array([Z0, phi0], dtype=complex), vectorized=True,
                              args=args_ws2, rtol=1e-08, atol=1e-10)
        Z2, phi2 = solution2.y[0, :], solution2.y[1, :]
        if plot_Zphi:
            plt.plot(np.real(Z2), np.imag(Z2))
            plt.show()
        theta_transformed_2.append(np.angle(disk_automorphism_bounded(Z2[-1], phi2[-1], z2_t)))

    theta_transformed_1 = np.array(theta_transformed_1)
    theta_transformed_2 = np.array(theta_transformed_2)

    theta_ws1 = np.where(theta_transformed_1 < 0, 2 * np.pi + theta_transformed_1, theta_transformed_1)
    theta_ws2 = np.where(theta_transformed_2 < 0, 2 * np.pi + theta_transformed_2, theta_transformed_2)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 7))
    ax1.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[2], label="Source")
    ax1.set_ylabel("Phase")   # $\\theta_1(t), ..., \\theta_N(t)$")
    ax1.set_ylim([-0.05, 2*np.pi + 0.05])
    ax1.legend()

    theta0_transformed = np.concatenate([np.array([theta0[0]]), theta_ws1[0, :], theta_ws2[0, :]])
    theta_verif = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0_transformed, *args_dynamics))
    theta_verif = np.where(theta_verif < 0, 2*np.pi + theta_verif, theta_verif)

    ax2.plot(timelist, theta[:, 1] % (2*np.pi), color=deep[0], label="Periphery 1")
    ax2.plot(timelist, theta[:, 2:1+sizes_crossratio[0]] % (2*np.pi), color=deep[0])
    ax2.plot(timelist, theta_verif[:, 1] % (2*np.pi), color=deep[7], label="True")
    ax2.plot(timelist, theta_verif[:, 2:1+sizes_crossratio[0]] % (2*np.pi), color=deep[7])
    ax2.plot(timelist, theta_ws1[:, 0] % (2*np.pi), color=deep[1], linestyle="--", label="Theory (transf. periphery 1)")
    ax2.plot(timelist, theta_ws1[:, 1:] % (2*np.pi), color=deep[1], linestyle="--")
    ax2.set_ylim([-0.05, 2*np.pi + 0.05])
    ax2.set_title(f"$\\epsilon_1 =$ {epsilon1}")
    ax2.set_ylabel("Phase")
    ax2.legend(loc=1)

    ax3.plot(timelist, theta[:, 1+sizes_crossratio[0]] % (2*np.pi), color=deep[0], label="Periphery 2")
    ax3.plot(timelist, theta[:, 2+sizes_crossratio[0]:] % (2*np.pi), color=deep[0])
    ax3.plot(timelist, theta_verif[:, 1+sizes_crossratio[0]] % (2*np.pi), color=deep[7], label="True")
    ax3.plot(timelist, theta_verif[:, 2+sizes_crossratio[0]:] % (2*np.pi), color=deep[7])
    ax3.plot(timelist, theta_ws2[:, 0] % (2*np.pi), color=deep[1], linestyle="--", label="Theory (transf. periphery 2)")
    ax3.plot(timelist, theta_ws2[:, 1:] % (2*np.pi), color=deep[1], linestyle="--")
    ax3.set_ylim([-0.05, 2*np.pi + 0.05])
    ax3.set_title(f"$\\epsilon_2 =$ {epsilon2}")
    ax3.set_ylabel("Phase")
    ax3.set_xlabel("Time $t$")
    ax3.legend(loc=1)

    w1 = omega[0]
    w = omega[1]
    Ws = W[1, 0]
    alphas = alpha[1, 0]
    conserved_monomial = np.exp(1j*theta[:, 0])*np.exp(-1j*w1*timelist)
    logc1234 = log_cross_ratio_theta(theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3])
    logc2345 = log_cross_ratio_theta(theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4])
    d2_2logc2345 = np.sin((theta[:, 3] - theta[:, 4])/2) / \
                   (np.sin((theta[:, 3] - theta[:, 1])/2)*np.sin((theta[:, 4] - theta[:, 1])/2))
    S2_2logc2345 = (w - w1 + coupling*Ws*np.sin(theta[:, 0] - theta[:, 1] - alphas))*d2_2logc2345
    # S2_alpha0 = np.cos((theta[:, 0] - theta[:, 1])/2)*np.sin((theta[:, 0] - theta[:, 1])/2)*d2_2logc2345
    d3_2logc2345 = np.sin((theta[:, 4] - theta[:, 3])/2) / \
                   (np.sin((theta[:, 3] - theta[:, 2])/2)*np.sin((theta[:, 4] - theta[:, 2])/2))
    S3_2logc2345 = (w - w1 + coupling*Ws*np.sin(theta[:, 0] - theta[:, 2] - alphas))*d3_2logc2345

    # plt.plot(timelist, np.real(conserved_monomial), label="Re($z_1e^{-i\\omega_1 t}$)")
    # plt.plot(timelist, np.imag(conserved_monomial), label="Im($z_1e^{-i\\omega_1 t}$)")
    plt.plot(timelist, np.real(conserved_monomial) + np.imag(conserved_monomial),
             label="Monomial: Re($z_1e^{-i\\omega_1 t}$) + Im($z_1e^{-i\\omega_1 t}$)")
    plt.plot(timelist, logc1234, label="Cross-ratio: ln($c_{1234}$)")
    plt.plot(timelist, logc2345, label="Cross-ratio: ln($c_{2345}$)")
    plt.plot(timelist, S2_2logc2345, label="Symmetry-generated: $\\mathcal{S}_2(2\\ln(c_{2345}))$")
    # plt.plot(timelist, S2_alpha0, label="Symmetry-generated: $\\mathcal{S}_2((2/\\sigma_1)\\ln(c_{2345}))$")
    plt.plot(timelist, S3_2logc2345, label="Symmetry-generated: $\\mathcal{S}_3(2\\ln(c_{2345}))$")
    plt.xlabel("Time $t$")
    # plt.xticks([0, 5, 10, 15, 20])
    plt.legend(title="Constants of motion", frameon=True, facecolor='white', edgecolor='0.7',
               framealpha=1, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=fontsize_legend)
    plt.show()

