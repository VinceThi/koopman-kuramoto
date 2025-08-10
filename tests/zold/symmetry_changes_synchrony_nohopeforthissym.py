# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
import matplotlib.pyplot as plt

# TODO seed for the whole script ?
from dynamics.symmetries import ode_symmetry_action_calS, disk_automorphism_bounded
from dynamics.constants_of_motion import cross_ratio_theta
from tqdm import tqdm
from scipy.integrate import solve_ivp
from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45
from dynamics.synchronization import kuramoto_order_parameter
import time
import json
from pathlib import Path
import sys
import random


# tseed = random.randrange(sys.maxsize)
# trng = random.Random(seed)
# tprint("Seed was:", seed)

random.seed(1351196896876584777)      # Special behavior

""" Partition parameters"""
sizes_monomial = np.array([1], dtype=int)
sizes_crossratio = np.array([4, 93], dtype=int)
size_nonintegrable = np.array([], dtype=int)
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
q = len(sizes)  # Number of parts
N = np.sum(sizes, dtype=int)
m = len(sizes_monomial)
c = len(sizes_crossratio)


""" Get weight matrix parameters """
x = np.zeros((sum(sizes_monomial), 1))
x[0, 0] = 1.0
random_exponents = np.concatenate([x, np.random.normal(1, 0.5, (sum(sizes_monomial), 1))], axis=1)
probabilities_monomial = np.array([0])
probabilities_crossratio = np.array([[1, 0, 0],
                                     [1, 0., 0.3]])
probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
weights_monomial = np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))

meanW = 0
stdW = 1
weights_crossratio = np.random.normal(meanW, stdW, (len(sizes_crossratio), N))
weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

coupling = 1


""" Get phase-lag matrix """
probabilities_monomial2 = np.array([0])
probabilities_crossratio2 = np.array([[1, 0, 0],
                                      [1, 0., 0.6]])
probabilities_nonintegrable2 = np.array([])
probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
meanalpha = 0
stdalpha = 1
phaselags_monomial = np.random.normal(meanalpha, stdalpha, (sum(sizes_monomial), sum(sizes_monomial))) % (np.pi/2)
phaselags_crossratio = np.random.normal(meanalpha, stdalpha, (len(sizes_crossratio), N))
phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
alpha, chi = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                     probabilities=probabilities_dict2, phaselags=phaselags_dict)


""" Initial conditions and integration parameters """
N = len(alpha[0])
theta0 = np.random.uniform(0, 2*np.pi, N)


t0, t1, dt = 0, 100, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))

source_crpart_strength_array = np.linspace(0.01, 10, 10)
R_forward = np.zeros((len(source_crpart_strength_array), len(timelist)))
for i, source_crpart_strength in tqdm(enumerate(source_crpart_strength_array)):
    weights_crossratio[1, 0] = source_crpart_strength
    W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                                probabilities=probabilities_dict, weights=weights_dict)
    cal_A = calA(coupling, C, chi)
    omega = random_gaussian_frequencies_pintegrable(m, c, sizes, cal_A, 0.5, 1)
    omega[0] = 0  # Equivalent to being in the frame of the source
    args_dynamics = (W, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics)) % (2*np.pi)
    # theta0 = theta[-1, :]
    R_forward[i, :] = kuramoto_order_parameter(theta[:, 5:])
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.plot(timelist, theta[:, 0])
    plt.plot(timelist, theta[:, 5:], color=deep[1])
    plt.subplot(212)
    plt.plot(timelist, R_forward[i, :])
    plt.show()

R_backward = np.zeros((len(source_crpart_strength_array), len(timelist)))
for j, source_crpart_strength in tqdm(enumerate(source_crpart_strength_array[::-1])):
    weights_crossratio[1, 0] = source_crpart_strength
    W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                                probabilities=probabilities_dict, weights=weights_dict)
    cal_A = calA(coupling, C, chi)
    omega = random_gaussian_frequencies_pintegrable(m, c, sizes, cal_A, 0.5, 1)
    omega[0] = 0  # Equivalent to being in the frame of the source
    args_dynamics = (W, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
    theta = np.where(theta < 0, 2*np.pi + theta, theta)
    # theta0 = theta[-1, :]
    R_backward[j, :] = kuramoto_order_parameter(theta[:, 5:])

plt.plot(source_crpart_strength_array, np.sum(R_forward, axis=1)/len(timelist), label="R_forward")
plt.plot(source_crpart_strength_array, np.sum(R_backward[::-1], axis=1)/len(timelist), label="R_backward")
plt.legend()
plt.show()
