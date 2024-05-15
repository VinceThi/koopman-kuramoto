# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz import ws_equations_kuramoto, ws_transformation
from dynamics.symmetries import *
from scipy.integrate import solve_ivp
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions

plot_trajectories = True

""" Graph parameters """
N = 4
W = np.ones((N, N))

""" Dynamical parameters """
t0, t1, dt = 0, 20, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega = 1
coupling = 1/N
theta0 = np.array([0, 2, 4, 6])

""" Integrate Kuramoto model """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Integrate Watanabe-Strogatz equations """
Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N)

args_ws = (w, coupling, omega)
solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
Z, phi = solution[:, 0], solution[:, 1]

theta_ws = []
for i in range(len(timelist)):
    theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
theta_ws = np.array(theta_ws)

""" Integrate determining equations """
Zbar0 = np.conjugate(Z0)
# sol_det = np.array(integrate_dopri45(t0, t1, dt, infinitesimal_condition_symmetry_kuramoto,
#                                      np.array([Z0, Zbar0, phi0]), *args_ws))
sol_det = solve_ivp(infinitesimal_condition_symmetry_kuramoto, [t0, t1], np.array([Z0, Zbar0, phi0], dtype=complex),
                    "BDF", args=args_ws, rtol=1e-8, atol=1e-12, vectorized=True)
Zdet, Zbardet, phidet = sol_det[:, 0], sol_det[:, 1], sol_det[:, 2]

""" Plot trajectories """
plt.figure(figsize=(10, 10))

plt.subplot(311)
theta = theta % (2*np.pi)
theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)
plt.plot(timelist, theta, color=deep[0])  # , label="Kuramoto")
plt.plot(timelist, theta_ws, color=deep[1], linestyle="--")  # , label="WS")
plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
plt.xlabel("Time $t$")

plt.subplot(312)
plt.plot(np.real(Z), np.imag(Z), label="WS")
plt.plot(np.real(Zdet), np.imag(Zdet), label="Det. eq.")
plt.plot(np.real(Zbardet), -np.imag(Zbardet), label="Det. eq. (conj(Zbar))")
plt.ylabel("Im Z(t)")
plt.xlabel("Re Z(t)")
plt.legend()

plt.subplot(313)
plt.plot(timelist, phi, label="WS")
plt.plot(timelist, phidet, label="Det. eq.")
plt.ylabel("$\\phi(t)$")
plt.xlabel("Time $t$")
plt.legend()

plt.show()
