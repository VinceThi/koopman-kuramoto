import numpy as np
from scipy.integrate import solve_ivp
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz import ws_equations_kuramoto, ws_transformation
from dynamics.symmetries import symmetry_generator_coefficients
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from plots.config_rcparams import *
import pytest


def test_symmetry_coefficients_kuramoto():

    plot_trajectories = True
    use_solve_ivp = False

    """ Parameters """
    t0, t1, dt = 0, 5, 0.01
    """ Important observation regarding the timestep dt """
    """ If dt is chosen very small (e.g., 0.001 or 0.0001), DOPRI45 will produce larger and larger errors in the
    in the integration"""
    timelist = np.linspace(t0, t1, int(t1 / dt))
    if use_solve_ivp:
        t_span = [t0, t1]
        integration_method = "BDF"
        rtol = 1e-8
        atol = 1e-12
    alpha = 0
    N = 10
    W = np.ones((N, N))
    omega = 1
    coupling = 5/N
    np.random.seed(499)
    theta0 = np.random.uniform(0, 2*np.pi, N)

    """ Integrate complete dynamics """
    args_dynamics = (W, coupling, omega, alpha)
    if use_solve_ivp:
        sol = solve_ivp(kuramoto_sakaguchi, t_span, theta0, integration_method,
                        args=args_dynamics, rtol=rtol, atol=atol, vectorized=True)
        theta = sol.y.T
        tc = sol.t
    else:
        theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))
    z = np.exp(1j*theta)

    """ Integrate reduced dynamics """
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N, nb_guess=5000)
    args_ws = (w, coupling, omega)
    if use_solve_ivp:
        solution = solve_ivp(ws_equations_kuramoto, t_span, np.array([Z0, phi0], dtype="complex"), integration_method,
                             args=args_ws, rtol=rtol, atol=atol)
        sol = solution.y.T
        Z, phi = sol[:, 0], sol[:, 1]
        tr = solution.t
    else:
        solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
        Z, phi = solution[:, 0], solution[:, 1]

    """ Compute coefficients """
    p1t = []
    for i in range(len(z[:, 0])):
        p1t.append(coupling/2*np.sum(z[i, :]))

    Lm1_coefficients = []
    L0_coefficients = []
    L1_coefficients = []
    for i in range(len(Z)):
        Lm1_coefficient, L0_coefficient , L1_coefficient \
            = symmetry_generator_coefficients(Z[i], phi[i], w, coupling, omega)
        Lm1_coefficients.append(Lm1_coefficient)
        L0_coefficients.append(L0_coefficient)
        L1_coefficients.append(L1_coefficient)

    p1t = np.array(p1t)
    Lm1_coefficients = np.array(Lm1_coefficients)
    L0_coefficients = np.array(L0_coefficients)
    L1_coefficients = np.array(L1_coefficients)

    if plot_trajectories:
        plt.figure(figsize=(6, 6))
        angle = np.linspace(0, 2*np.pi, 1000)
        plt.plot(N*coupling/2*np.cos(angle), N*coupling/2*np.sin(angle), linewidth=5, color=total_color,
                 label="Upper bound on $p_n$")
        plt.plot(np.real(p1t), np.imag(p1t), color=deep[3], label="$p_1(t)$")
        plt.plot(np.real(Lm1_coefficients), np.imag(Lm1_coefficients), color=deep[6], linestyle="--",
                 label="$L_{-1}$ coefficient")
        plt.scatter([0], [omega], s=40, color=deep[0], label="$i\\omega$")
        plt.scatter(np.real(L0_coefficients), np.imag(L0_coefficients), color=deep[9], linestyle="--",
                    label="$L_0$ coefficient", s=5)
        plt.ylabel("Im")
        plt.xlabel("Re")
        plt.legend()
        plt.show()

    assert np.all(np.abs(Lm1_coefficients - p1t) < 1e-4) and np.all(np.abs(L0_coefficients - 1j*omega) < 1e-4)


if __name__ == "__main__":
    pytest.main()
