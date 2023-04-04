# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import networkx as nx
from plots.config_rcparams import *
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.constants_of_motion import cross_ratio_theta
from dynamics.integrate import integrate_dopri45
import time
import json
import tkinter.simpledialog
from tkinter import messagebox

""" Time parameters """
t0, t1, dt = 0, 30, 0.1
timelist = np.linspace(t0, t1, int(t1 / dt))

""" Graph parameters """
graph_str = "star"
N = 5
G = nx.star_graph(N-1)
W = nx.to_numpy_array(G)


""" Dynamical parameters """
dynamics_str = "kuramoto_sakaguchi"
omega = np.array([0.5, 1, 1, 1, 1])
coupling = 1
alpha = 0

""" Integration """
x0 = np.random.random(N)

args_dynamics = (coupling, omega, alpha)
x = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi,
                               W, x0, *args_dynamics))

fig = plt.figure(figsize=(4, 4))
plt.subplot(111)
for j in range(0, N):
    plt.plot(timelist, x[:, j]%(2*np.pi), color=first_community_color,
             linewidth=0.3)
plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 2], x[:, 3], x[:, 4]))
# plt.plot(cross_ratio_theta(x[:, 2], x[:, 3], x[:, 4], x[:, 2]))
ylab = plt.ylabel('$\\theta_j$', labelpad=20)
ylab.set_rotation(0)
plt.xlabel('Time $t$')
plt.tick_params(axis='both', which='major')
plt.tight_layout()
# plt.legend(loc=4, fontsize=fontsize_legend)
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

    parameters_dictionary = {"graph_str": graph_str, "N": N,
                             "omega": omega.tolist(),
                             "alpha": alpha,
                             "coupling": coupling,
                             "t0": t0, "t1": t1, "dt": dt}

    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
                f'_{dynamics_str}_{graph_str}.pdf')
    fig.savefig(path + f'{timestr}_{file}_bifurcation_diagram'
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
