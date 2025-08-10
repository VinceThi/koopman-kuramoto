# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.symmetries import ode_symmetry_action_calS, disk_automorphism_bounded
from dynamics.constants_of_motion import cross_ratio_theta
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

plot_trajectories = False
import_setup = True
simulate = False

if import_setup:
    def load_parameters_to_globals(filepath, keys_as_array=None):
        """
        Load a parameter dictionary from JSON and unpack into globals().
        Converts specified list-valued keys back into NumPy arrays.

        Parameters:
            filepath (str or Path): path to the JSON file
            keys_as_array (list): keys that should be converted back to np.array
        """
        filepath = Path(filepath)
        with filepath.open("r") as f:
            params = json.load(f)

        keys_as_array = set(keys_as_array or [])
        for key, val in params.items():
            if key in keys_as_array and isinstance(val, list):
                val = np.array(val)
            globals()[key] = val
    SCRIPT_DIR = Path(__file__).resolve().parent   # Get current script location
    REPO_ROOT = SCRIPT_DIR.parent      # Go to repo root (adjust this based on how deep your script is)
    path = REPO_ROOT / 'simulations' / 'kooku1_fig3_data'  # Path to the data file
    # Path to your JSON file
    file_path = Path(path / "2025_08_04_13h15min51sec_higher_value_cross_ratio_kuramoto_parameters_dictionary.json")
    load_parameters_to_globals(file_path, keys_as_array=[
        "sizes_monomial", "sizes_crossratio", "size_nonintegrable",
        "W", "alpha", "omega", "C", "chi", "theta0", "theta",
        "epsilon_array", "random_exponents",
        "probabilities_monomial", "probabilities_crossratio",
        "probabilities_nonintegrable2", "probabilities_monomial2", "probabilities_crossratio2", "timelist"])
    cal_A = calA(coupling, C, chi)
    # for i in range(len(theta345_transformed)):
    #     plt.plot(np.array(theta345_transformed)[i, :, :], color=np.random.rand(3,))
    # plt.show()
    d2_2logc2345 = np.sin((theta0[3] - theta0[4])/2) / \
                   (np.sin((theta0[3] - theta0[1])/2)*np.sin((theta0[4] - theta0[1])/2))
    S2_2logc2345 = (omega[1] + coupling*W[1, 0]*np.sin(theta0[0] - theta0[1] - alpha[1, 0]))*d2_2logc2345
    print(S2_2logc2345)
    plt.plot(timelist, np.array(theta345_transformed)[0, :, 1])
    plt.plot(timelist, np.array(theta345_transformed)[0, :, 2])
    plt.show()

else:
    """ Partition parameters"""
    sizes_monomial = np.array([1], dtype=int)
    sizes_crossratio = np.array([4, 4], dtype=int)
    size_nonintegrable = np.array([], dtype=int)
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
    q = len(sizes)  # Number of parts
    N = np.sum(sizes, dtype=int)
    m = len(sizes_monomial)
    c = len(sizes_crossratio)

    """ Get weight matrix """
    random_exponents = np.array([1])
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
    probabilities_nonintegrable2 = np.array([])
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
    # We put ourselves in the reference frame of the source
    omega[0] = 0
    # We have that omega[1 to 4] = omega[0] - 2*np.imag(cal_A[0, 0]) to ensure c1234 is conserved, so we set :
    omega[1] = omega[0] - 2*np.imag(cal_A[0, 0])
    omega[2] = omega[0] - 2*np.imag(cal_A[0, 0])
    omega[3] = omega[0] - 2*np.imag(cal_A[0, 0])
    omega[4] = omega[0] - 2*np.imag(cal_A[0, 0])
    Omega1 = omega[1] - 2*np.imag(cal_A[0, 1])
    Omega2 = omega[5] - 2*np.imag(cal_A[1, 5])
    print("omega = ", omega)
    print("Omega1 = ", Omega1)
    print("Omega2 = ", Omega2)

    """ Initial conditions and integration parameters """
    N = len(alpha[0])
    # np.random.seed(12)
    theta0 = np.random.uniform(0, 2*np.pi, N)
    theta0[0] = 0
    theta0[1] = 4.7
    theta0[2] = 3.16
    theta0[3] = 0.5014
    print("theta0 = ", theta0)


if not import_setup or simulate:
    t0, t1, dt = 0, 10, 0.001
    timelist = np.linspace(t0, t1, int(t1 / dt))

    args_dynamics = (W, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
    theta = np.where(theta < 0, 2*np.pi + theta, theta)
    c1234_t_eps0 = cross_ratio_theta(0, theta[:, 1], theta[:, 2], theta[:, 3])
    # Ensure that the cross-ratio is conserved
    assert np.all(np.abs(c1234_t_eps0-c1234_t_eps0[0])<1e-2)
    
    """ Constants of motion """
    # Conserved cross-ratio c1234
    cross_ratio_vs_epsilon = [cross_ratio_theta(theta0[0], theta0[1], theta0[2], theta0[3])]
    
    # Conserved symmetry-generated constant of motion calS_2[c2345]
    w1 = omega[0]
    w = omega[1]
    Ws = W[1, 0]
    alphas = alpha[1, 0]
    d2_2logc2345 = np.sin((theta0[3] - theta0[4])/2) / \
                   (np.sin((theta0[3] - theta0[1])/2)*np.sin((theta0[4] - theta0[1])/2))
    S2_2logc2345 = (w - w1 + coupling*Ws*np.sin(theta0[0] - theta0[1] - alphas))*d2_2logc2345
    sym_gen_vs_epsilon = [S2_2logc2345]
    
    
    """ Action of the symmetry on the cross-ratio c_1234 and calS_2[c_2345] """
    epsilon_array = np.arange(0, 5.5, 0.5)   # starts at zero np.linspace(0, 2, 50)
    theta2 = theta[:, 1]
    theta345_transformed_epsilon = [theta[:, 2:1 + sizes_crossratio[0]].tolist()]
    for epsilon in tqdm(epsilon_array[1:]):
        """ Integrate symmetry """
        Z0, phi0 = 0, 0

        theta_transformed = []
        for t in range(len(timelist)):
            zs_t = np.exp(1j * theta[t, 0])

            # We consider the symmetry calS_3 + calS_4 + calS_5 and it only acts on z_3, z_4, z_5 (indices 2,3,4)
            z_t = np.exp(1j * theta[t, 2:1 + sizes_crossratio[0]])

            args_ws1 = (z_t, cal_A[0, 0], cal_A[0, 2:1 + sizes_crossratio[0]], Omega1 - omega[0], zs_t)
            solution = solve_ivp(ode_symmetry_action_calS, [0, epsilon],
                                 np.array([Z0, phi0], dtype=complex), vectorized=True,
                                 args=args_ws1, rtol=1e-08, atol=1e-10)
            Z, phi = solution.y[0, :], solution.y[1, :]
            t_trans = np.angle(disk_automorphism_bounded(Z[-1], phi[-1], z_t))
            theta_transformed.append(t_trans.tolist())
        theta345_transformed_epsilon.append(theta_transformed)
        theta_transformed = np.array(theta_transformed)
        theta_transformed = np.where(theta_transformed < 0, 2 * np.pi + theta_transformed, theta_transformed)
        theta3_trans, theta4_trans, theta5_trans = theta_transformed[:, 0], theta_transformed[:, 1], theta_transformed[:, 2]

        c1234_t = cross_ratio_theta(0, theta2, theta3_trans, theta4_trans)    
        # print("c1234_t = ", c1234_t)

        # Ensure that the cross-ratio is conserved
        assert np.all(np.abs(c1234_t-c1234_t[0]) < 1e-2)
        cross_ratio_vs_epsilon.append(c1234_t[0])
    
        d2_2logc2345_t = np.sin((theta4_trans - theta5_trans) / 2) / \
                        (np.sin((theta4_trans - theta2) / 2) * np.sin((theta5_trans - theta2) / 2))
        S2_2logc2345_t = (w - w1 + coupling * Ws * np.sin(theta0[0] - theta2 - alphas))*d2_2logc2345_t

        # Ensure that the sym-gen constant of motion is conserved
        assert np.all(np.abs(S2_2logc2345_t-S2_2logc2345_t[0]) < 1e-2)
        sym_gen_vs_epsilon.append(S2_2logc2345_t[0])
        if plot_trajectories:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            ax1.plot(timelist, theta[:, 0] % (2 * np.pi), color=deep[2], label="Source")
            ax1.set_ylabel("Phase")  # $\\theta_1(t), ..., \\theta_N(t)$")
            ax1.set_ylim([-0.05, 2 * np.pi + 0.05])
            ax1.legend()
    
            theta0_transformed = np.concatenate([np.array([theta0[0], theta0[1]]), theta_transformed[0, :], theta0[5:]])
            theta_verif = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0_transformed, *args_dynamics))
            theta_verif = np.where(theta_verif < 0, 2 * np.pi + theta_verif, theta_verif)
    
            ax2.plot(timelist, theta[:, 1] % (2 * np.pi), color=deep[0], label="Periphery 1")
            ax2.plot(timelist, theta[:, 1:1 + sizes_crossratio[0]] % (2 * np.pi), color=deep[0])
            ax2.plot(timelist, theta_verif[:, 2] % (2 * np.pi), color=deep[7], label="True")
            ax2.plot(timelist, theta_verif[:, 3:1 + sizes_crossratio[0]] % (2 * np.pi), color=deep[7])
            ax2.plot(timelist, theta_transformed[:, 0] % (2 * np.pi), color=deep[1], linestyle="--",
                     label="Theory (transf. periphery 1)")
            ax2.plot(timelist, theta_transformed[:, 1:] % (2 * np.pi), color=deep[1], linestyle="--")
            ax2.set_ylim([-0.05, 2*np.pi + 0.05])
            c_verif = np.round(cross_ratio_theta(0, theta2[0], theta3_trans[0], theta4_trans[0]), 3)
            ax2.set_title(f"$\\epsilon =$ {np.round(epsilon, 2)}, c1234 = {c_verif}")
            ax2.set_ylabel("Phase")
            ax2.legend(loc=1)
    
            plt.show()


fig = plt.figure(figsize=(8, 5))

ax1 = fig.add_subplot(211)
ax1.scatter(epsilon_array, cross_ratio_vs_epsilon, s=10)
ax1.set_ylabel("Cross-ratio $c_{1234}($exp$(\\varepsilon\\mathcal{S})z)$")
ax1.set_xlabel("$\\varepsilon$")

ax2 = fig.add_subplot(212)
ax2.scatter(epsilon_array, sym_gen_vs_epsilon, s=10)
ax2.set_ylabel("$\\mathcal{S}_2[\\ln c_{2345}($exp$(\\varepsilon\\mathcal{S})z)]$")
ax2.set_xlabel("$\\varepsilon$")

plt.show()
if not import_setup or simulate:
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
        parameters_dictionary = {"N": N, "sizes_monomial": sizes_monomial.tolist(),
                                 "sizes_crossratio":sizes_crossratio.tolist(),
                                 "size_nonintegrable":size_nonintegrable.tolist(), "W": W.tolist(),
                                 "alpha": alpha.tolist(), "coupling": coupling,
                                 "omega": omega.tolist(), "Omega1": Omega1, "Omega2": Omega2,
                                 "C": C.tolist(), "chi": chi.tolist(), "t0": t0, "t1": t1, "dt": dt,
                                 "theta": theta.tolist(),
                                 "meanW": meanW, "stdW": stdW, "meanalpha": meanalpha, "stdalpha": stdalpha,
                                 "theta0": theta0.tolist(), "random_exponents": random_exponents.tolist(),
                                 "probabilities_monomial": probabilities_monomial.tolist(),
                                 "probabilities_crossratio": probabilities_crossratio.tolist(),
                                 "probabilities_nonintegrable2": probabilities_nonintegrable2.tolist(),
                                 "probabilities_monomial2": probabilities_monomial2.tolist(),
                                 "probabilities_crossratio2": probabilities_crossratio2.tolist(),
                                 "epsilon_array": epsilon_array.tolist(), "cross_ratio_vs_epsilon":cross_ratio_vs_epsilon,
                                 "sym_gen_vs_epsilon":sym_gen_vs_epsilon,
                                 "theta345_transformed": theta345_transformed_epsilon,
                                 "timelist": timelist.tolist()
                                 }
        fig.savefig(path / f'{timestr}_{filename}_symmetry_change_invariant_sets_kuramoto.pdf')
        fig.savefig(path / f'{timestr}_{filename}_symmetry_change_invariant_sets_kuramoto.png')
        with open(path / f'{timestr}_{filename}_kuramoto_parameters_dictionary.json', 'w') as outfile:
            json.dump(parameters_dictionary, outfile)
