import numpy as np
from dynamics.watanabe_strogatz import ws_transformation, Z_dot


def nu_function(R, Y):
    X = np.sqrt(R**2 - Y**2 + 1)
    Gamma = np.sqrt(X**2 - 1)
    ratio = (X + Gamma) / (X - Gamma)
    return np.log(ratio) / (2 * Gamma)


def determining_equations_real_disk_automorphism(t, state, theta, current_index, omega, coupling):
    R, Phi, Y = state
    p0 = len(theta[0, :])*coupling/2
    p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
    p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
    rho1, phi1 = np.abs(p1), np.angle(p1)
    rho2, phi2 = np.abs(p2), np.angle(p2)
    chi1 = 2*rho1*np.sin(Phi - phi1)
    chi2 = p0 - rho2*np.cos(2*Phi - phi2)
    mu = (1 - nu_function(R, Y)*np.sqrt(R**2 - Y**2 + 1))/(R**2 - Y**2)*(chi1*Y*R + chi2*R**2)
    dRdt = (chi2 - mu)*R
    dPhidt = omega + rho2*np.sin(2*Phi - phi2)
    dYdt = -mu*Y - chi1*R
    return np.array([dRdt, dPhidt, dYdt])




def rfun(Z, Zbar, phi):
    zeta = np.exp(-1j*phi)
    x = 1 + zeta
    y = np.sqrt((1 - zeta)**2 + 4*zeta*Z*Zbar)
    ratio = (x + y)/(x - y)
    module = np.abs(ratio)
    argument = np.angle(ratio)
    return (np.log(module) + 1j*argument)/y
    # return np.log((x + y)/(x - y))/y


def infinitesimal_condition_symmetry_kuramoto(t, state, p1, current_index, omega):
    Z, Zbar, phi = state
    r = rfun(Z, Zbar, phi)
    normZ2 = Z*Zbar
    zeta = np.exp(-1j*phi)
    zetabar = np.exp(1j*phi)
    gamma = (1 - zeta)**2 + 4*zeta*Z*Zbar
    alpha = (1 - zetabar - r*zeta + r - 2*r*Z*Zbar)/gamma
    beta = ((1 + zeta)/(1 - Z*Zbar) - 2*r*zeta)/gamma
    k = current_index
    pm1 = np.conjugate(p1[k])

    A = np.array([[r + alpha*(zeta-1) + beta*normZ2, - beta*Zbar**2, beta*(1 - zeta)*Zbar],
                  [-beta*zetabar*Z**2, r + alpha*(zeta - 1) + beta*zetabar*normZ2, beta*(1 - zeta)*Z],
                  [-(r*zetabar + alpha + beta*zetabar*normZ2)*Z, (beta*zetabar*normZ2 - alpha)*Zbar, r+2*beta*normZ2]])
    b = np.array([1j*omega*Z + (1-zetabar)*p1[k], -1j*omega*Zbar + (1-zeta)*pm1, 2*pm1*Z*zeta - 2*p1[k]*Zbar])
    dZdt, dZbardt, dzetadt = (1-normZ2)*zeta*(A@b)
    return np.array([dZdt, dZbardt, np.angle(dzetadt)])


def infinitesimal_condition_symmetry_kuramoto_2(t, state, p1, current_index, omega):
    X, Y, phi = state
    Z = X + 1j*Y
    Zbar = X - 1j*Y
    normZ2 = X**2 + Y**2
    r = rfun(Z, Zbar, phi)
    zeta = np.exp(-1j*phi)
    zetabar = np.exp(1j*phi)
    gamma = (1 - zeta)**2 + 4*zeta*normZ2
    alpha = (1 - zetabar - r*zeta + r - 2*r*normZ2)/gamma
    beta = ((1 + zeta)/(1 - normZ2) - 2*r*zeta)/gamma
    k = current_index
    pm1 = np.conjugate(p1[k])

    A = np.array([[1j*beta*(1 - zetabar)*X*Y + r + alpha*(zeta - 1) + beta*(1 + zetabar)*Y**2,
                   beta*X*(1j*X*(zetabar-1)-Y*(zetabar+1)),
                   2*beta*(1 - zeta)*X],
                  [beta*Y*(1j*Y*(1 - zetabar) - X*(zetabar + 1)),
                   1j*beta*(zetabar - 1)*X*Y + r + alpha*(zeta - 1) + beta*(1 + zetabar)*X**2,
                   2*beta*(1 - zeta)*Y],
                  [-2*alpha*X - r*zetabar*(X + 1j*Y) - 2*1j*beta*zetabar*Y*normZ2,
                   -2*alpha*Y + 1j*r*zetabar*(X + 1j*Y) + 2*1j*beta*zetabar*X*normZ2,
                   2*r+4*beta*normZ2]])
    b = np.array([-omega*Y + ((1-zetabar)*p1[k] + (1-zeta)*pm1)/2,
                  omega*X + ((1-zetabar)*p1[k] - (1-zeta)*pm1)/(2*1j),
                  (pm1*zeta - p1[k])*X + 1j*(pm1*zeta + p1[k])*Y])
    dXdt, dYdt, dzetadt = (1-normZ2)*zeta*(A@b)
    return np.array([dXdt, dYdt, np.angle(dzetadt)])


def determining_equations_disk_automorphism(t, state, theta, current_index, omega, coupling):
    V, Y = state
    Vbar = np.conjugate(V)
    p0 = len(theta[0, :])*coupling/2
    p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
    p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
    dVdt = (p0 + 1j*omega)*V - p2*Vbar
    dYdt = 1j*(np.conjugate(p1)*V - p1*Vbar)
    return np.array([dVdt, dYdt])


# def nu_function(R, Y, Phi):
#     Gamma = np.sqrt(X**2 - 1)
#     ratio = (X + Gamma)/(X - Gamma)
#     return np.log(ratio)/(2*Gamma)
#
#
# def mu_function(X, nu):
#     return (1 - nu*X)/(X**2 - 1)
#
#
# def determining_equations_real_disk_automorphism(t, state, theta, current_index, omega, coupling):
#     R, Phi, X, Y = state
#     nu = nu_function(X)
#     mu = mu_function(X, nu)
#     p0 = len(theta[0, :])*coupling/2
#     p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
#     p2 = coupling/2*np.sum(np.exp(2*1j*theta[current_index, :]))
#     rho1, phi1 = np.abs(p1), np.angle(p1)
#     rho2, phi2 = np.abs(p2), np.angle(p2)
#     chi1 = 2*rho1*np.sin(2*Phi - phi1)
#     chi2 = p0 - rho2*np.cos(2*Phi - phi2)
#     dRdt = chi2*R - mu*chi1*R**2 - mu*chi2*R**3
#     dPhidt = omega + rho2*np.sin(2*Phi - phi2)
#     dXdt = nu*(chi1*Y*R + chi2*R**2)
#     dYdt = -chi1*R - mu*(chi1*R*Y**2 + chi2*R**2*Y)
#     return np.array([dRdt, dPhidt, dXdt, dYdt])
#


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