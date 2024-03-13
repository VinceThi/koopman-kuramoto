import numpy as np
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz import ws_equations_kuramoto, ws_transformation
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from plots.config_rcparams import *
import pytest


def test_ws_equations_kuramoto():

    print("\nBeginning test_ws_equations_kuramoto...")
    plot_trajectories = False

    """ Parameters """
    t0, t1, dt = 0, 10, 0.005
    timelist = np.linspace(t0, t1, int(t1 / dt))
    alpha = 0
    N = 100
    W = np.ones((N, N))
    omega = 1
    coupling = 50/N
    np.random.seed(499)
    theta0 = np.random.uniform(0, 2*np.pi, N)

    """ Integrate complete dynamics """
    args_dynamics = (W, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics)) % (2*np.pi)

    """ Integrate reduced dynamics """
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N, nb_guess=10000)
    args_ws = (w, coupling, omega)
    solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
    Z, phi = solution[:, 0], solution[:, 1]
    theta_ws = []
    for i in range(len(timelist)):
        theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)

    if plot_trajectories:
        plt.figure(figsize=(6, 6))
        plt.plot(timelist, theta[:, 0], color=deep[0], label="original system")
        plt.plot(timelist, theta[:, 1:], color=deep[0])
        plt.plot(timelist, theta_ws[:, 0], color=deep[1], linestyle="--", label="watanabe-strogatz")
        plt.plot(timelist, theta_ws[:, 1:], color=deep[1], linestyle="--")
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
        plt.xlabel("Time $t$")
        plt.legend()
        plt.show()

    assert np.all(np.abs(theta - theta_ws) < 1e-6)


test_ws_equations_kuramoto()
if __name__ == "__main__":
    pytest.main()
