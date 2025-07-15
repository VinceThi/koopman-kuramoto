# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
import matplotlib.pyplot as plt

from plots.config_rcparams import *
import numpy as np
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45
from dynamics.synchronization import kuramoto_order_parameter  #, network_order_parameter
from tqdm import tqdm

plot_time_series = False

""" Define dynamical parameters """
w1, w2, w3, w4, w5 = 1, 0., -0.598, 0., 1.512
W =  np.array([[0., 0., 0., 0., 0.],
               [w1, 0., w3, w4, w5],
               [w1, w2, 0., w4, w5],
               [w1, w2, w3, 0., w5],
               [w1, w2, w3, w4, 0.]])
# W = ((W + W.T) > 0)
a1, a2, a3, a4, a5 =  0.1, -0.7, np.pi/2-0.1, 0.9, 1  #  0., 0., 0., 0., 0.   #
alpha =  np.array([[0., 0., 0., 0., 0.],
                   [a1, 0., a3, a4, a5],
                   [a1, a2, 0., a4, a5],
                   [a1, a2, a3, 0., a5],
                   [a1, a2, a3, a4, 0.]])
print("W = \n", W)
print("alpha = \n", alpha)
# omega1, omega2 = 2.1345, 0.493341
# omega =  [omega1, omega2, omega2, omega2, omega2]


""" Integration parameters """
t0, t1, dt = 0, 200, 0.05
timelist = np.linspace(t0, t1, int(t1 / dt))
time_mean_index = int((t1//dt)//3)
N = len(alpha[0])
nb_coupling = 10
nb_init = 5
nb_instances = 1
coupling_array = np.linspace(0.1, 4, nb_coupling)
initcond_array = np.linspace(0.1, np.pi, nb_init)

R_global = np.zeros((nb_coupling, nb_init))
R_periphery = np.zeros((nb_coupling, nb_init))
for i, coupling in tqdm(enumerate(coupling_array)):
    # If the phase lags are not zero:
    calA = coupling/2*np.array([[w1*np.exp(1j*a1), w2*np.exp(1j*a2), w3*np.exp(1j*a3),
                                 w4*np.exp(1j*a4), w5*np.exp(1j*a5)]])
    omega1, omega2 = 2.1345, 0.493341
    omega = [omega1, omega2, omega2 + 2*np.imag(calA[0, 2] - calA[0, 1]), omega2 + 2*np.imag(calA[0, 3] - calA[0, 1]), omega2 + 2*np.imag(calA[0, 4] - calA[0, 1])]
    for j, delta in tqdm(enumerate(initcond_array)):
        #print(coupling, delta)
        R_global_instance = []
        R_periphery_instance = []
        for _ in range(nb_instances):
            theta0 = np.random.uniform(-delta, delta, N)
            # print("init cond = ", theta0)
            # print("omega = ", omega)

            """ Integrate """
            args_dynamics = (W, coupling, omega, alpha)
            theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))
            R_global_instance.append(np.mean(kuramoto_order_parameter(theta)[-time_mean_index:]))
            R_periphery_instance.append(np.mean(kuramoto_order_parameter(theta[:, 1:])[-time_mean_index:]))
            if plot_time_series:
                """ Illustration of the results"""
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))

                ax1.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[0], linewidth=1)
                ax1.set_ylabel("Source's phase")
                ax1.set_xlim(timelist[0], timelist[-1])
                ax1.set_ylim(0, 2*np.pi)

                for k in range(4):
                    ax2.plot(timelist, theta[:, 1+k] % (2*np.pi), color=deep[1], linewidth=1)
                ax2.set_ylabel("Peripheral phases")
                ax2.set_xlim(timelist[0], timelist[-1])
                ax2.set_ylim(0, 2*np.pi)

                ax3.plot(timelist, kuramoto_order_parameter(theta[:, 1:]), color=dark_grey, label="R_{\\mathrm{kur}}")
                # ax3.plot(timelist, network_order_parameter(theta[:, 1:], coupling, W, alpha), color=reduced_grey, label="R_{\\mathrm{net}}")
                ax3.set_ylabel("Peripheral synchro")
                ax3.set_xlim(timelist[0], timelist[-1])
                ax3.set_ylim(-0.05, 1.05)

                ax4.plot(timelist, kuramoto_order_parameter(theta), color=dark_grey)
                # ax4.plot(timelist, network_order_parameter(theta, coupling, W, alpha), color=reduced_grey, label="R_{\\mathrm{net}}")
                ax4.set_xlim(timelist[0], timelist[-1])
                ax4.set_ylim(-0.05, 1.05)
                ax4.set_ylabel("Global synchro")
                ax4.set_xlabel("Time $t$")

                plt.tight_layout()
                plt.show()
        R_global[i, j] = np.mean(np.array(R_global_instance))
        R_periphery[i, j] = np.mean(np.array(R_periphery_instance))

if nb_coupling == 1:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(initcond_array, R_global[0])
    ax.set_ylabel("Synchronization measure")
    ax.set_xlabel("Initial conditions range")
    plt.show()
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(R_periphery, aspect="auto",
                     origin="lower",
                     extent=[initcond_array[0], initcond_array[-1], coupling_array[0], coupling_array[-1]])
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel("Initial conditions range")
    ax1.set_ylabel("Coupling")
    ax1.set_title("Peripheral synchrony", pad=15)
    plt.tight_layout()

    im2 = ax2.imshow(R_global, aspect="auto",
                     origin="lower",
                     extent=[initcond_array[0], initcond_array[-1], coupling_array[0], coupling_array[-1]])
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel("Initial conditions range")
    ax2.set_ylabel("Coupling")
    ax2.set_title("Global synchrony", pad=15)
    plt.tight_layout()
    plt.show()