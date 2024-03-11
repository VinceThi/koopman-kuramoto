# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import os
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi


""" Weight matrix """
directory = "../plots/integrability_partitioned_graph"
if not os.path.exists(directory):
    os.makedirs(directory)
file_path = os.path.join(directory, '2024_01_13_15h38min26sec_integrability_partitioned_block_weight_matrix.npy')

W = np.load(file_path)
sizes = [38, 4, 58, 150, 250]
N = np.sum(sizes)
n = len(sizes)

""" Dynamical parameters """
t0, t1, dt = 0, 50, 0.05
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
omega_non_integrable = np.random.uniform(-1, 1, sizes[0])
Omega_integrable = [-0.8, 1, -0.5, 0.1]
omega = omega_non_integrable.tolist()
for i, size in enumerate(sizes[1:]):
    if size == sizes[0]:
        omega += omega_non_integrable.tolist()
    else:
        omega += size*[Omega_integrable[i]]
coupling = 1   # np.linspace(0.1, 1, 50)
theta0 = np.random.uniform(0, 2*np.pi, N)
# np.concatenate([np.random.uniform(0, 2*np.pi, 42), np.ones(56),
#  np.array([0.8]), np.array([1.2]), np.random.uniform(0, 2*np.pi, 400)])

""" Integrate """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))

""" Measure synchro """


def uniform_reduction_matrix(sizes):
    M = np.zeros((n, N))
    delta = np.diag(np.ones(len(sizes)))
    q = len(sizes)
    for mu in range(q):
        row_blocks = []
        for nu in range(q):
            row_blocks.append(delta[mu][nu]*np.ones((1, sizes[nu]))/sizes[mu])
        if not mu:
            M = np.block(row_blocks)
        else:
            row_blocks = np.concatenate(row_blocks, axis=1)
            M = np.concatenate([M, row_blocks], axis=0)
    return M


M = uniform_reduction_matrix(sizes)
r_mu_array = np.zeros((len(theta[:, 0]), n))
for mu in range(n):
    r_mu = np.absolute(np.sum(M[mu, :] * np.exp(1j*theta), axis=1))
    r_mu_array[:, mu] = r_mu

# for i in range(42, 100):
#     plt.plot(timelist, theta[:, i]%(2*np.pi))
colors = ["#C44E52", "#8172B3", "#DD8452",  "#55A868", "#4C72B0"]
for mu in range(n):
    plt.plot(timelist, r_mu_array[:, mu], color=colors[mu])
plt.ylabel("Group synchronization parameter $R_\\mu$")
plt.xlabel("Time $t$")
plt.show()


# if __name__ == "__main__":
#     sizes = [38, 4, 58, 150, 250]
