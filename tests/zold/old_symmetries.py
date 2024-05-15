import numpy as np
from dynamics.watanabe_strogatz import ws_transformation, Z_dot


# def infinitesimal_condition_symmetry_kuramoto(t, state, w, coupling, omega):
#     Z, phi = state
#     F_bar = coupling / (2 * 1j) * np.sum(ws_transformation(Z, phi, w))
#     G = omega
#     F = np.conjugate(F_bar)
#
#     dphidt = F*Z + 2*F_bar*np.conjugate(Z) + (np.exp(1j*phi) - 1)*F_bar/Z + F*Z*np.exp(1j*phi)
#     return np.array([Z_dot(Z, F, G, F_bar), np.real(dphidt)])


def rfun(Z, Zbar, phi):
    zeta = np.exp(-1j * phi)
    x = 1 + zeta
    y = np.sqrt((1 - zeta) ** 2 + 4 * zeta * Z * Zbar)
    return np.log((x + y) / (
                x - y)) / y  # np.log or cmath.log gives the same thing


def watanabe_strogatz_generator_on_w(w, Z, phi):
    Zbar = np.conjugate(Z)
    zeta = np.exp(-1j * phi)
    return rfun(Z, Zbar, phi) * (zeta * Z + (1 - zeta) * w - Zbar * w ** 2)


def infinitesimal_condition_symmetry_kuramoto(t, state, w, coupling, omega):
    Z, Zbar, phi = state
    r = rfun(Z, Zbar, phi)
    p1 = coupling / 2 * np.sum(ws_transformation(Z, phi, w))
    pm1 = np.conjugate(p1)
    zeta = np.exp(-1j * phi)
    zetabar = np.exp(1j * phi)
    gamma = (1 - zeta) ** 2 + 4 * zeta * Z * Zbar
    alpha = (1 - zetabar - r * zeta + r - 2 * r * Z * Zbar) / gamma
    beta = ((1 + zeta) / (1 - Z * Zbar) - 2 * r * zeta) / gamma
    print(Z[0], r, p1, pm1, zeta, gamma, alpha, beta)

    A = np.array([[r + alpha * (zeta - 1) + beta * Z * Zbar, - beta * Zbar ** 2, beta * (1 - zeta) * Zbar],
                  [-beta * zetabar * Z ** 2, r + alpha * (zeta - 1) + beta * zetabar * Z * Zbar, beta * (1 - zeta) * Z],
                  [-(r * zetabar + alpha + beta * zetabar * Z * Zbar) * Z, (beta * zetabar * Z * Zbar - alpha) * Zbar,
                   r + 2 * beta * Z * Zbar]],
                 dtype=complex).squeeze()
    b = np.array([1j * omega * Z + (1 - zetabar) * p1, -1j * omega * Zbar + (1 - zeta) * pm1,
                  2 * pm1 * Z * zeta - 2 * p1 * Zbar], dtype=complex)
    print(np.shape(A), np.shape(b))
    print(phi, w)
    return (1 - Z * Zbar) * zeta * (A @ b)


def infinitesimal_condition_symmetry_kuramoto_simplified(t, state, w, coupling, omega):
    Z, phi = state
    F_bar = coupling / (2 * 1j) * np.sum(ws_transformation(Z, phi, w))
    G = omega
    F = np.conjugate(F_bar)

    dphidt = F * Z + 2 * F_bar * np.conjugate(Z) + (np.exp(1j * phi) - 1) * F_bar / Z + F * Z * np.exp(1j * phi)
    return np.array([Z_dot(Z, F, G, F_bar), np.real(dphidt)])


def test_ws_equations_kuramoto_symmetry():
    plot_trajectories = True

    """ Parameters """
    t0, t1, dt = 0, 10, 0.001
    timelist = np.linspace(t0, t1, int(t1 / dt))
    alpha = 0
    N = 10
    W = np.ones((N, N))
    omega = 1
    coupling = 2 / N
    np.random.seed(499)
    theta0 = np.random.uniform(0, 2 * np.pi, N)

    """ Integrate complete dynamics """
    args_dynamics = (W, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics)) % (2 * np.pi)

    """ Integrate reduced dynamics """
    Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N, nb_guess=5000)
    args_ws = (w, coupling, omega)
    solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
    Z, phi = solution[:, 0], solution[:, 1]
    theta_ws = []
    for i in range(len(timelist)):
        theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2 * np.pi + theta_ws, theta_ws)

    phi = np.where(phi < 0, 2 * np.pi + phi, phi)
    phi = np.where(phi > 2 * np.pi, phi - 2 * np.pi, phi)

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
    phi_inf = np.where(phi_inf > 2 * np.pi, phi_inf - 2 * np.pi, phi_inf)

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