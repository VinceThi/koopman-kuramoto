# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import os
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
import time


""" Weight matrix """
directory = "/home/vincent/git_repositories/koopman-kuramoto/plots/integrability_partitioned_graph"
if not os.path.exists(directory):
    os.makedirs(directory)
file_path = os.path.join(directory, '2024_01_13_15h38min26sec_integrability_partitioned_block_weight_matrix.npy')

W = np.load(file_path)
sizes = [38, 4, 58, 150, 250]
N = np.sum(sizes)

""" Dynamical parameters """
t0, t1, dt = 0, 100, 0.2
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega_non_integrable = np.random.uniform(-1, 1, sizes[0])
Omega_integrable = [1, -0.8, 0.8, 0.5, 0.1]
omega = []
for size, i in enumerate(sizes):
    if size == sizes[0]:
        omega += omega_non_integrable.tolist()
    else:
        omega += size*[Omega_integrable[i]]
coupling = 1   # np.linspace(0.1, 1, 50)
x0 = np.random.uniform(0, 2*np.pi, N)

""" Integrate """
args_dynamics = (coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, W, x0, *args_dynamics))

""" Measure synchro """
M
synchro_order_parameter = np.absolute(np.sum(M[mu, :]*np.exp(1j*theta), axis=1))
