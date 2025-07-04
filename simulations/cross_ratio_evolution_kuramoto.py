# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from plots.config_rcparams import *
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.constants_of_motion import cross_ratio_theta,\
    log_cross_ratio_theta
from dynamics.integrate import integrate_dopri45
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

""" Time parameters """
t0, t1, dt = 0, 15, 0.0001
timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "motif"

# Directed star (periphery to core)
# W = np.array([[0, 1, 1, 1, 1],
#               [0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0]])

# Directed star (core to periphery)
# W = np.array([[0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0]])
w = 1
W = np.array([[0, 0, 0, 0],
              [w, 0, 0, 0],
              [w, 0, 0, 0],
              [w, 0, 0, 0]])

# Undirected star
# W = np.array([[0, 1, 1, 1, 1],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0]])

# Complete graph
# W = np.ones((5, 5))

# Other motif
# W = np.array([[0, 1, 1, 1, 1],
#               [1, 0, 1, 0, 1],
#               [1, 0, 1, 0, 1],
#               [1, 0, 1, 0, 1],
#               [1, 0, 1, 0, 1]])

# Directed star in network (periphery to core)
# W = np.array([[0, 0, 0, 0, 0, 0, 0],
#               [1, 0, 1, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, 0]])
# omega = np.array([0.5, 0.7, 0.8,  1, 1, 1, 1])

# W = np.array([[0, 1, 1, 1, 1, 1],
#               [1, 0, 1, 1, 1, 1],
#               [0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0]])
# omega = np.array([0.5, 0.7, 1, 1, 1, 1])

# W = np.array([[0, 1, 1, 1, 1],
#               [0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0],
#               [0, 1, 0, 0, 0],
#               [0, 1, 0, 0, 0]])

# W = np.array([[0, 1, 1, 1, 1],
#               [0, 0, 1, 0, 0],
#               [0, 1, 0, 0, 0],
#               [0, 1, 1, 0, 0],
#               [0, 1, 1, 0, 0]])


N = len(W[0])

""" Dynamical parameters """
dynamics_str = "kuramoto_sakaguchi"
coupling = 0.5/5
alpha = np.pi/3
As = (coupling/2)*w*np.exp(-1j*alpha)
omega = np.array([1 + 2*np.imag(As), 1, 1, 1])

""" Integration """
theta0 = 2*np.pi*np.random.random(N)
args_dynamics = (W, coupling, omega, alpha)
x = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))


fig = plt.figure(figsize=(4, 4))
plt.subplot(111)
for j in range(0, N):
    plt.plot(timelist, x[:, j] % (2*np.pi), color=first_community_color,
             linewidth=0.3)
# plt.plot(timelist, log_cross_ratio_theta(theta0[1], theta0[2], theta0[3], theta0[4])*np.ones(len(timelist)),
#          label="Log cross-ratio $log(c_{2345})$")
# plt.plot(timelist, log_cross_ratio_theta(x[:, 1], x[:, 2], x[:, 3], x[:, 4]))
plt.plot(timelist, log_cross_ratio_theta(x[:, 0], x[:, 1], x[:, 2], x[:, 3]))
# plt.plot(timelist, cross_ratio_theta(theta0[1], theta0[2], theta0[3], theta0[4])*np.ones(len(timelist)))
# plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 2], x[:, 3], x[:, 4]))  # label="Cross-ratio $c_{2345}$")
#          label="Cross-ratio $log(c_{2345})$")
# plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 2], x[:, 4], x[:, 3]))
# plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 3], x[:, 2], x[:, 4]))
# plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 3], x[:, 4], x[:, 2]))
# plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 4], x[:, 2], x[:, 3]))
# plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 4], x[:, 3], x[:, 2]))
# plt.plot(timelist, cross_ratio_theta(x[:, 0], x[:, 1], x[:, 2], x[:, 3]))
ylab = plt.ylabel('$\\theta_j$', labelpad=20)
ylab.set_rotation(0)
plt.xlabel('Time $t$')
# plt.ylim([-1, 2*np.pi + 1])
plt.tick_params(axis='both', which='major')
plt.tight_layout()
plt.legend(loc=1, fontsize=fontsize_legend)
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters,"
                       " the data, and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    path = f'C:/Users/thivi/Documents/GitHub/koopman-kuramoto/' \
           f'simulations/data/{dynamics_str}_data/'
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    parameters_dictionary = {"N": N, "graph_str": graph_str,
                             "omega": omega.tolist(),
                             "alpha": alpha,
                             "coupling": coupling,
                             "t0": t0, "t1": t1, "dt": dt}

    fig.savefig(path + f'{timestr}_{file}_trajectories_and_cross_ratios'
                f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_trajectories_and_cross_ratios'
                       f'_{dynamics_str}_{graph_str}.png')
    # with open(path + f'{timestr}_{file}'
    #           f'_x_equilibrium_points_list'
    #           f'_complete_{dynamics_str}_{graph_str}.json', 'w') \
    #         as outfile:
    #     json.dump(x_equilibrium_points_list, outfile)
    # with open(path + f'{timestr}_{file}'
    #           f'_redx_equilibrium_points_list'
    #           f'_reduced_{dynamics_str}_{graph_str}.json',
    #           'w') as outfile:
    #     json.dump(redx_equilibrium_points_list, outfile)
    with open(path + f'{timestr}_{file}'
              f'_{dynamics_str}_{graph_str}_parameters_dictionary.json',
              'w') as outfile:
        json.dump(parameters_dictionary, outfile)
