# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto


""" Parameters """
N = 2
omega = np.array([0.5, 1])
W = np.array([[0, 0.5],
              [1, 0]])
alpha = np.array([[0, np.pi/3],
                  [-np.pi/3, 0]])

t0, t1, dt = 0, 100, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))
coupling = 1
theta0 = np.array([0, np.pi-0.2])  # np.random.uniform(0, 2*np.pi, N)


""" Integrate """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))

plt.plot(timelist, theta[:, 0] % (2*np.pi))
plt.plot(timelist, theta[:, 1] % (2*np.pi))
plt.ylabel("Phases")
plt.xlabel("Time $t$")
plt.show()
