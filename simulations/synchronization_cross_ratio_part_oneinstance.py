# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from dynamics.watanabe_strogatz import ws_transformation
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from dynamics.integrate import integrate_dopri45
from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from plots.kuramoto_animation import animate_kuramoto_on_circle
from matplotlib.animation import FuncAnimation
from dynamics.dynamics import kuramoto
from numba import njit
import time
import json
from pathlib import Path
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog

verif_validity_ws_equations = False
import_setup = False
isolate_cross_ratio_part = import_setup*False
reconnect_cross_ratio_part = import_setup*False
plot_animation = True

@njit(fastmath=True)
def ws_transformation(Z, phi, w):
    eiphi = np.exp(1j*phi)
    num = eiphi*w + Z
    den = 1.0 + eiphi*np.conjugate(Z)*w
    return num / den

@njit(fastmath=True)
def ws_equations_kooku1_fig3(t, state, theta_series, current_index,
                             w, calA_sources, calA_row_periphery, omega):
    """
    Hardcoded for Fig. 3 (a)
      state = [Z, phi] (complex128)
      theta_series[i] -> theta at time step i (float64 vector of size N)
      theta_series is assumed shape (n_steps, N); we use current_index 'i' to take the row.
      w, calA_sources, calA_row_periphery are vectors (see below)
      omega is real scalar
    Returns a tuple (dotZ, dotphi) to avoid per-call array allocation.
    """
    Z = state[0]
    phi = state[1]

    theta = theta_series[current_index]    # shape (N,)
    zt = np.exp(1j * theta)                # complex (N,)

    sources_input = np.sum(calA_sources * zt[:3])

    F = np.sum(calA_row_periphery * ws_transformation(Z, phi, w)) + sources_input
    G = omega
    F_bar = np.conjugate(F)

    dotZ   = F + 1j * G * Z - F_bar * Z * Z
    dotphi = G - 1j * F * np.conjugate(Z) + 1j * F_bar * Z
    return dotZ, dotphi   # tuple (faster / no heap alloc)

@njit(fastmath=True)
def integrate_dopri45_non_autonomous(t0, t1, dt, dynamics, init_cond,
                                     non_autonomous_term,
                                     w, calA_sources, calA_row_periphery, omega):
    """
    dynamics(t, y, non_autonomous_term, i, w, calA_sources, calA_row_periphery, omega)
        -> returns (dotZ, dotphi)  # a tuple of two complex128 scalars
    init_cond: complex128 array of shape (2,)  [Z0, phi0]
    non_autonomous_term: e.g. theta_series with shape (n_steps, N)
    w, calA_sources, calA_row_periphery: vectors (dtypes consistent with complex math)
    omega: float64
    """
    n_steps = int((t1 - t0) / dt)
    Y = np.empty((n_steps, 2), dtype=np.complex128)
    Y[0, 0] = init_cond[0]
    Y[0, 1] = init_cond[1]

    # stage buffers
    k1 = np.empty(2, np.complex128)
    k2 = np.empty(2, np.complex128)
    k3 = np.empty(2, np.complex128)
    k4 = np.empty(2, np.complex128)
    k5 = np.empty(2, np.complex128)
    k6 = np.empty(2, np.complex128)
    ytmp = np.empty(2, np.complex128)

    for i in range(n_steps):
        t = t0 + i * dt
        y0 = Y[i]

        dz, dphi = dynamics(t, y0, non_autonomous_term, i, w, calA_sources, calA_row_periphery, omega)
        k1[0] = dz;      k1[1] = dphi

        ytmp[0] = y0[0] + dt*(1.0/5.0)*k1[0]
        ytmp[1] = y0[1] + dt*(1.0/5.0)*k1[1]
        dz, dphi = dynamics(t + (1.0/5.0)*dt, ytmp, non_autonomous_term, i, w, calA_sources, calA_row_periphery, omega)
        k2[0] = dz;      k2[1] = dphi

        ytmp[0] = y0[0] + dt*((3.0/40.0)*k1[0] + (9.0/40.0)*k2[0])
        ytmp[1] = y0[1] + dt*((3.0/40.0)*k1[1] + (9.0/40.0)*k2[1])
        dz, dphi = dynamics(t + (3.0/10.0)*dt, ytmp, non_autonomous_term, i, w, calA_sources, calA_row_periphery, omega)
        k3[0] = dz;      k3[1] = dphi

        ytmp[0] = y0[0] + dt*((44.0/45.0)*k1[0] - (56.0/15.0)*k2[0] + (32.0/9.0)*k3[0])
        ytmp[1] = y0[1] + dt*((44.0/45.0)*k1[1] - (56.0/15.0)*k2[1] + (32.0/9.0)*k3[1])
        dz, dphi = dynamics(t + (4.0/5.0)*dt, ytmp, non_autonomous_term, i, w, calA_sources, calA_row_periphery, omega)
        k4[0] = dz;      k4[1] = dphi

        ytmp[0] = y0[0] + dt*((19372.0/6561.0)*k1[0] - (25360.0/2187.0)*k2[0] +
                              (64448.0/6561.0)*k3[0] - (212.0/729.0)*k4[0])
        ytmp[1] = y0[1] + dt*((19372.0/6561.0)*k1[1] - (25360.0/2187.0)*k2[1] +
                              (64448.0/6561.0)*k3[1] - (212.0/729.0)*k4[1])
        dz, dphi = dynamics(t + (8.0/9.0)*dt, ytmp, non_autonomous_term, i, w, calA_sources, calA_row_periphery, omega)
        k5[0] = dz;      k5[1] = dphi

        ytmp[0] = y0[0] + dt*((9017.0/3168.0)*k1[0] - (355.0/33.0)*k2[0] +
                              (46732.0/5247.0)*k3[0] + (49.0/176.0)*k4[0] -
                              (5103.0/18656.0)*k5[0])
        ytmp[1] = y0[1] + dt*((9017.0/3168.0)*k1[1] - (355.0/33.0)*k2[1] +
                              (46732.0/5247.0)*k3[1] + (49.0/176.0)*k4[1] -
                              (5103.0/18656.0)*k5[1])
        dz, dphi = dynamics(t + dt, ytmp, non_autonomous_term, i, w, calA_sources, calA_row_periphery, omega)
        k6[0] = dz;      k6[1] = dphi

        Y[i+1, 0] = y0[0] + dt*((35.0/384.0)*k1[0] + (500.0/1113.0)*k3[0] +
                                (125.0/192.0)*k4[0] - (2187.0/6784.0)*k5[0] +
                                (11.0/84.0)*k6[0])
        Y[i+1, 1] = y0[1] + dt*((35.0/384.0)*k1[1] + (500.0/1113.0)*k3[1] +
                                (125.0/192.0)*k4[1] - (2187.0/6784.0)*k5[1] +
                                (11.0/84.0)*k6[1])
    return Y


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
    file_path = Path(path / "2025_08_21_15h03min45sec_periodic_kuramoto_parameters_dictionary.json")  # "2025_08_21_15h29min15sec_phasesync_kuramoto_parameters_dictionary.json")  #
    load_parameters_to_globals(file_path, keys_as_array=[
        "sizes_monomial", "sizes_crossratio", "size_nonintegrable",
        "theta0", "random_exponents", "W", "alpha", "C", "chi", "omega", "Rew", "Imw", "ReZ0", "ImZ0",
        "probabilities_monomial", "probabilities_crossratio",
        "probabilities_nonintegrable2", "probabilities_monomial2", "probabilities_crossratio2", "timelist"])
    w = np.array(Rew) + 1j*np.array(Imw)
    Z0 = np.array(ReZ0) + 1j*np.array(ImZ0)
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable]).astype(int)

else:  # Generate parameters
    """ Integration parameters """
    t0, t1, dt = 0, 300, 0.01
    timelist = np.linspace(t0, t1, int(t1 / dt))

    """ Partition parameters"""
    # Note: This does not yield the graph from fig. 3, but it does yields the cross-ratio part of 93 vertices
    # that receive the input from the source 1
    sizes_monomial = np.array([1, 2], dtype=int)
    sizes_crossratio = np.array([4, 93], dtype=int)
    size_nonintegrable = np.array([], dtype=int)
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
    q = len(sizes)  # Number of parts
    N = int(np.sum(sizes))
    m = len(sizes_monomial)
    c = len(sizes_crossratio)

    """ Coupling """
    coupling = 1

    """ Distribution parameters """

    # Weight matrix
    random_exponents = np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
    probabilities_monomial = np.array([0, 1])
    probabilities_crossratio = np.array([[1, 0, 0, 0],
                                         [1, 1, 0., 0.8]])
    probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
    meanW_monomial = 1     # Conformist pair
    stdW_monomial = 0.1
    weights_monomial = np.random.normal(meanW_monomial, stdW_monomial, (sum(sizes_monomial), sum(sizes_monomial)))

    weights_crossratio10 = 0     # TODO Important parameters
    weights_crossratio11 = 0      # TODO Important parameters
    weights_crossratio12 = 0      # TODO Important parameters

    meanW_crossratio = 0.1  # -0.2
    stdW_crossratio = 1

    weights_crossratio = np.random.normal(meanW_crossratio, stdW_crossratio, (len(sizes_crossratio), N))
    weights_crossratio[1, 0] = weights_crossratio10
    weights_crossratio[1, 1] = weights_crossratio11
    weights_crossratio[1, 2] = weights_crossratio12
    weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

    W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                                probabilities=probabilities_dict, weights=weights_dict)

    # Phase-lags
    probabilities_monomial2 = np.array([0, 1])
    probabilities_crossratio2 = np.array([[1, 0, 0, 0],
                                          [1, 1, 0., 0.5]])
    probabilities_nonintegrable2 = np.array([])
    probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
    meanalpha_monomial = 0
    stdalpha_monomial = 1
    meanalpha_crossratio = 1
    stdalpha_crossratio = 0.2

    phaselags_monomial = np.random.normal(meanalpha_monomial, stdalpha_monomial,
                                          (sum(sizes_monomial), sum(sizes_monomial))) % (np.pi / 2)
    phaselags_crossratio = np.random.normal(meanalpha_crossratio, stdalpha_crossratio, (len(sizes_crossratio), N))
    phaselags_crossratio[1, 0] = 0   # TODO Important parameters
    print(phaselags_crossratio[1, 0])
    phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
    alpha, chi = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                         probabilities=probabilities_dict2, phaselags=phaselags_dict)

    """ calA """
    cal_A = calA(coupling, C, chi)

    # Natural frequencies
    mean_omega, std_omega = 0.5, 1
    omega = random_gaussian_frequencies_pintegrable(m, c, sizes, cal_A, mean_omega, std_omega)
    # We put ourselves in the reference frame of the source
    omega[0] = 0
    # We have that omega[1 to 4] = omega[0] - 2*np.imag(cal_A[0, 0]) to ensure c1234 is conserved, so we set :
    # Not important for this script however, since we focus on the other cross-ratio part
    omega[3] = omega[0] - 2*np.imag(cal_A[0, 0])
    omega[4] = omega[0] - 2*np.imag(cal_A[0, 0])
    omega[5] = omega[0] - 2*np.imag(cal_A[0, 0])
    omega[6] = omega[0] - 2*np.imag(cal_A[0, 0])
    Omega1 = omega[3] - 2*np.imag(cal_A[0, 3])
    Omega2 = omega[7] - 2*np.imag(cal_A[1, 7])

    # print("W = \n", np.round(W, 3))
    # # plt.matshow(W, aspect="auto")
    # # plt.show()
    # print("alpha = \n", np.round(alpha, 3))
    # print("calA = \n", np.round(cal_A, 3))
    # print("conformist_contrarian_coupling = \n", conformist_contrarian_coupling)
    # # print("omega = ", omega)
    # print("Omega1 = ", Omega1)
    # print("Omega2 = ", Omega2)
    # # print("theta0 = ", theta0)

    theta0 = np.random.uniform(0, 2*np.pi, N)
    theta0[0] = 0  # This simplifies things, without loss of generality
    percentage_averaged_end_time_series = 0.5
    start_idx = int(percentage_averaged_end_time_series*len(timelist))
        
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0[7:], sizes_crossratio[1], nb_guess=5000)


""" Simulation """
if isolate_cross_ratio_part:
    C[1, 0] = 0
    W[7:, 0] = 0
if reconnect_cross_ratio_part:
    C[1, 0] = -5
    W[7:, 0] = -5
    chi[1, 0] = 0.1
    alpha[7:, 0] = 0.1

cal_A = calA(coupling, C, chi)

conformist_contrarian_coupling = coupling * np.sum(C[1, 7:]*np.cos(chi[1, 7:])) / sizes_crossratio[1]  # from Lohe 2017

print(conformist_contrarian_coupling)

args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
theta = np.where(theta < 0, 2*np.pi + theta, theta)

if plot_animation:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ani_ax = animate_kuramoto_on_circle(theta, sizes, interval=10, ax=ax)
    plt.show()

""" Integrate the (large-size) cross-ratio part """
# print("Initial conditions WS obtained")
args_ws = (w, np.array([cal_A[1, 0], cal_A[1, 1], cal_A[1, 2]]), cal_A[1, 7:], Omega2)
solution = np.array(integrate_dopri45_non_autonomous(t0, t1, dt, ws_equations_kooku1_fig3,
                    np.array([Z0, phi0], dtype=complex), theta, *args_ws))
# print("Integration of WS equations complete")
Z = solution[:, 0]
phi = np.real(solution[:, 1])      # np.real for JSON serialization
# ReZ, ImZ = np.real(Z), np.imag(Z)  # for JSON serialization
# Rew, Imw = np.real(w), np.imag(w)  # for JSON serialization

zeta = Z*np.exp(-1j*phi)
mean_module_zeta = np.mean(np.abs(zeta[start_idx:]))

if mean_module_zeta  > 1.05:
    raise ValueError('Something went wrong in the integration (e.g., too large integration step) and '
                     'np.mean(np.abs(zeta[start_idx:])) > 1.05')


if verif_validity_ws_equations:
    theta_ws = []
    for i in range(len(timelist)):
        theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2 * np.pi + theta_ws, theta_ws)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(timelist, theta[:, :3] % (2 * np.pi), color=deep[2], label="Sources")
    ax1.set_ylabel(
        "Phase")  # $\\theta_1(t), ..., \\theta_N(t)$")
    ax1.set_ylim([-0.05, 2 * np.pi + 0.05])
    ax1.legend()

    ax2.plot(timelist, theta[:, 7:] % (2 * np.pi),
             color=deep[0])  # , label="Cross-ratio part")
    ax2.plot(timelist, theta_ws % (2 * np.pi), color=deep[1],
             linestyle="dashed")  # , label="Cross-ratio part")
    ax2.set_ylim([-0.05, 2 * np.pi + 0.05])
    ax2.set_ylabel("Phase")
    ax2.legend(loc=1)
    plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
ax1.set_aspect('equal')
angle = np.linspace(0, 2*np.pi, 300)
ax1.plot(np.cos(angle), np.sin(angle), color=reduced_grey, alpha=0.8)
ax1.plot(np.real(zeta), np.imag(zeta), color="#aac4ff", label=r"$Z(t) e^{-i\phi(t)}$")
ax1.scatter(np.real(zeta[0]), np.imag(zeta[0]), color="#aac4ff", s=3)
ax1.axis('off')
ax1.legend(loc=1)
ax2.plot(timelist, np.abs(zeta))
ax2.set_xlabel("Time $t$")
ax2.set_ylabel("$|Z(t) e^{-i\phi(t)}|$")
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
    parameters_dictionary = {"N": N, "sizes_monomial": sizes_monomial.tolist(),
                             "sizes_crossratio":sizes_crossratio.tolist(),
                             "size_nonintegrable":size_nonintegrable.tolist(), "coupling": coupling,
                             "meanW_monomial": meanW_monomial, "stdW_monomial": stdW_monomial,
                             "meanalpha_monomial": meanalpha_monomial, "stdalpha_monomial": stdalpha_monomial,
                             "meanW_crossratio": meanW_crossratio, "stdW_crossratio": stdW_crossratio,
                             "meanalpha_crossratio": meanalpha_crossratio, "stdalpha_crossratio": stdalpha_crossratio,
                             "theta0": theta0.tolist(), "W": W.tolist(), "alpha": alpha.tolist(),
                             "C": C.tolist(), "chi": chi.tolist(), "omega": omega.tolist(),
                             "random_exponents": random_exponents.tolist(),
                             "probabilities_monomial": probabilities_monomial.tolist(),
                             "probabilities_crossratio": probabilities_crossratio.tolist(),
                             "probabilities_nonintegrable2": probabilities_nonintegrable2.tolist(),
                             "probabilities_monomial2": probabilities_monomial2.tolist(),
                             "probabilities_crossratio2": probabilities_crossratio2.tolist(),
                             "t0": t0, "t1": t1, "dt": dt, "theta": theta.tolist(),
                             "timelist": timelist.tolist(), "conformist_contrarian_coupling": conformist_contrarian_coupling,
                             "mean_module_zeta": mean_module_zeta,
                             "ReZ0": np.real(Z0).tolist(), "ImZ0": np.imag(Z0).tolist(), "phi0": np.real(phi0),
                             "Rew": np.real(w).tolist(), "Imw": np.imag(w).tolist(),
                             "ReZ": np.real(Z).tolist(), "ImZ": np.imag(Z).tolist(), "phi": np.real(phi).tolist(),
                             "Rezeta": np.real(zeta).tolist(), "Imzeta": np.imag(zeta).tolist(),
                             "percentage_averaged_end_time_series": percentage_averaged_end_time_series,
                             "start_idx": int(start_idx),
                             "mean_omega": mean_omega, "std_omega": std_omega, "Omega1": Omega1, "Omega2": Omega2
                             }

    fig.savefig(path / f'{timestr}_{filename}_synchro_crossratio_part_kuramoto.pdf')
    fig.savefig(path / f'{timestr}_{filename}_synchro_crossratio_part_kuramoto.png')
    with open(path / f'{timestr}_{filename}_kuramoto_parameters_dictionary.json', 'w') as outfile:
        json.dump(parameters_dictionary, outfile)
