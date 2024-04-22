# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi, complex_kuramoto_sakaguchi
from dynamics.symmetries import watanabe_strogatz_generator_on_w
from dynamics.watanabe_strogatz import ws_equations_kuramoto, ws_transformation
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
dotz = []
for i in range(len(timelist)):
    dotz.append(complex_kuramoto_sakaguchi(0, np.exp(1j*theta[i, :]), W, coupling, np.diag(omega*np.ones(N)), alpha))
dotz = np.array(dotz)

""" Integrate Watanabe-Strogatz equations """

# Arbitrary Z0, phi0, w
Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, N)

args_ws = (w, coupling, omega)
solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
Z, phi = solution[:, 0], solution[:, 1]

evaluated_generator = []
for i in range(len(timelist)):
    evaluated_generator.append(watanabe_strogatz_generator_on_w(w, Z[i], phi[i]))


""" Compare infinitesimal generators """
# Generate points on the unit circle
angle = np.linspace(0, 2*np.pi, 1000)

plt.figure(figsize=(5, 5))
plt.plot(np.cos(angle), np.sin(angle), color=complete_grey, linewidth=5)
plt.plot(np.real(dotz), np.imag(dotz), color=deep[0])
plt.plot(np.real(evaluated_generator), np.imag(evaluated_generator), color=deep[1], linestyle="--")
plt.xlabel("Real")
plt.ylabel("Imaginary")
# plt.xlim([-1.2, 1.2])
# plt.ylim([-1.2, 1.2])
plt.show()
