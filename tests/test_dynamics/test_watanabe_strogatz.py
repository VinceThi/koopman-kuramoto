import numpy as np
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz import ws_equations_kuramoto, ws_transformation
from dynamics.symmetries import infinitesimal_condition_symmetry_kuramoto_simplified
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from plots.config_rcparams import *
import pytest


# def test_ws_equations_kuramoto():
# 
#     plot_trajectories = True
# 
#     """ Parameters """
#     t0, t1, dt = 0, 10, 0.005
#     timelist = np.linspace(t0, t1, int(t1 / dt))
#     alpha = 0
#     N = 100
#     W = np.ones((N, N))
#     omega = 1
#     coupling = 50/N
#     np.random.seed(499)
#     theta0 = np.random.uniform(0, 2*np.pi, N)
# 
#     """ Integrate complete dynamics """
#     args_dynamics = (W, coupling, omega, alpha)
#     theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics)) % (2*np.pi)
# 
#     """ Integrate reduced dynamics """
#     Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N, nb_guess=5000)
#     args_ws = (w, coupling, omega)
#     solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
#     Z, phi = solution[:, 0], solution[:, 1]
#     theta_ws = []
#     for i in range(len(timelist)):
#         theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
#     theta_ws = np.array(theta_ws)
#     theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)
# 
#     if plot_trajectories:
#         plt.figure(figsize=(6, 6))
#         plt.plot(timelist, theta, color=deep[0])
#         plt.plot(timelist, theta_ws, color=deep[1], linestyle="--")
#         plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
#         plt.xlabel("Time $t$")
#         plt.show()
# 
#     assert np.all(np.abs(theta - theta_ws) < 1e-6)


def test_ws_equations_kuramoto_symmetry():

    plot_trajectories = True

    """ Parameters """
    t0, t1, dt = 0, 10, 0.001
    timelist = np.linspace(t0, t1, int(t1 / dt))
    alpha = 0
    N = 10
    W = np.ones((N, N))
    omega = 1
    coupling = 2/N
    np.random.seed(499)
    theta0 = np.random.uniform(0, 2*np.pi, N)

    """ Integrate complete dynamics """
    args_dynamics = (W, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics)) % (2*np.pi)

    """ Integrate reduced dynamics """
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N, nb_guess=5000)
    args_ws = (w, coupling, omega)
    solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
    Z, phi = solution[:, 0], solution[:, 1]
    theta_ws = []
    for i in range(len(timelist)):
        theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)

    phi = np.where(phi < 0, 2 * np.pi + phi, phi)
    phi = np.where(phi > 2*np.pi, phi - 2*np.pi, phi)

    """ Integrate infinitesimal condition of symmetry """
    solution_inf = np.array(integrate_dopri45(t0, t1, dt, infinitesimal_condition_symmetry_kuramoto_simplified,
                                              np.array([Z0, phi0]), *args_ws))
    Z_inf, phi_inf = solution_inf[:, 0], solution_inf[:, 1]
    theta_inf = []
    for i in range(len(timelist)):
        theta_inf.append(np.angle(ws_transformation(Z_inf[i], phi_inf[i], w)))
    theta_inf = np.array(theta_inf)
    theta_inf = np.where(theta_inf < 0, 2 * np.pi + theta_inf, theta_inf)
    
    phi_inf = np.where(phi_inf < 0, 2 * np.pi + phi_inf, phi_inf)
    phi_inf = np.where(phi_inf > 2*np.pi, phi_inf - 2*np.pi, phi_inf)

    if plot_trajectories:
        plt.figure(figsize=(8, 8))

        plt.subplot(411)
        plt.plot(timelist, theta, color=deep[0], linewidth=5)
        plt.plot(timelist, theta_ws, color=deep[1], linestyle="-")
        plt.plot(timelist, theta_inf, color=deep[2], linestyle="--", linewidth=1)
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")

        plt.subplot(412)
        # plt.plot(timelist, theta, color=deep[0])
        plt.plot(timelist, np.abs(Z), color=deep[1], linestyle="-")
        plt.plot(timelist, np.abs(Z_inf), color=deep[2], linestyle="--", linewidth=1)
        plt.ylabel("$|Z(t)|$")

        plt.subplot(413)
        # plt.plot(timelist, theta, color=deep[0])
        plt.plot(timelist, np.angle(Z), color=deep[1], linestyle="-")
        plt.plot(timelist, np.angle(Z_inf), color=deep[2], linestyle="--", linewidth=1)
        plt.ylabel("arg($Z(t)$)")

        plt.subplot(414)
        plt.plot(timelist, phi, color=deep[1], linestyle="-")
        plt.plot(timelist, phi_inf, color=deep[2], linestyle="--", linewidth=1)
        plt.ylabel("$\\varphi(t)$")
        plt.xlabel("Time $t$")
        plt.show()

    assert np.all(np.abs(theta - theta_ws) < 1e-6)


# test_ws_equations_kuramoto_symmetry()
if __name__ == "__main__":
    pytest.main()
