# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
import numpy as np
import matplotlib.pyplot as plt
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto
from dynamics.constants_of_motion import log_cross_ratio_theta


""" Parameters """
N = 5
alphas = 0  # np.pi/3
Ws = 0.5
coupling = 0.1
As = (coupling/2)*Ws*np.exp(-1j*alphas)
w = 1
w1 = w + 2*np.imag(As)
omega = np.array([w1, w, w, w, w])
W = np.array([[0, 0, 0, 0, 0],
              [Ws, 0, 0, 0, 0],
              [Ws, 0, 0, 0, 0],
              [Ws, 0, 0, 0, 0],
              [Ws, 0, 0, 0, 0]])
alpha = np.array([[0, 0, 0, 0, 0],
                  [alphas, 0, 0, 0, 0],
                  [alphas, 0, 0, 0, 0],
                  [alphas, 0, 0, 0, 0],
                  [alphas, 0, 0, 0, 0]])
t0, t1, dt = 0, 100, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
theta0 = np.random.uniform(0, 2*np.pi, N)   # np.array([0, np.pi/5, 2*np.pi/3, np.pi-0.1, 3*np.pi/2+0.1])  # np.random.uniform(0, 2*np.pi, N)
print("init cond = ", theta0)

""" Integrate """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))

""" Plot results"""

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "axes.labelsize": 14,
#     "font.size": 12,
#     "legend.fontsize": 12,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12
# })
fontsize_legend = 10

plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[0], label="Source (vertex 1)", linewidth=2)
plt.plot(timelist, theta[:, 1] % (2*np.pi), color=deep[1], label="Leaves (vertices 2,3,4,5)", linewidth=1)
plt.plot(timelist, theta[:, 2] % (2*np.pi), color=deep[1], linewidth=1)
plt.plot(timelist, theta[:, 3] % (2*np.pi), color=deep[1], linewidth=1)
plt.plot(timelist, theta[:, 4] % (2*np.pi), color=deep[1], linewidth=1)
plt.ylabel("Phases")
plt.xlabel("Time $t$")
plt.xticks([0, 5, 10, 15, 20])
plt.legend(frameon=True, facecolor='white', edgecolor='0.7',
           framealpha=1, loc='center left', bbox_to_anchor=(1.02, 0.5),
           fontsize=fontsize_legend)

plt.subplot(212)

conserved_monomial = np.exp(1j*theta[:, 0])*np.exp(-1j*w1*timelist)
logc1234 = log_cross_ratio_theta(theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3])
logc2345 = log_cross_ratio_theta(theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4])
d2_2logc2345 = np.sin((theta[:, 3] - theta[:, 4])/2) / \
               (np.sin((theta[:, 3] - theta[:, 1])/2)*np.sin((theta[:, 4] - theta[:, 1])/2))
S2_2logc2345 = (w - w1 + coupling*Ws*np.sin(theta[:, 0] - theta[:, 1] - alphas))*d2_2logc2345
# S2_alpha0 = np.cos((theta[:, 0] - theta[:, 1])/2)*np.sin((theta[:, 0] - theta[:, 1])/2)*d2_2logc2345
d3_2logc2345 = np.sin((theta[:, 4] - theta[:, 3])/2) / \
               (np.sin((theta[:, 3] - theta[:, 2])/2)*np.sin((theta[:, 4] - theta[:, 2])/2))
S3_2logc2345 = (w - w1 + coupling*Ws*np.sin(theta[:, 0] - theta[:, 2] - alphas))*d3_2logc2345

# plt.plot(timelist, np.real(conserved_monomial), label="Re($z_1e^{-i\\omega_1 t}$)")
# plt.plot(timelist, np.imag(conserved_monomial), label="Im($z_1e^{-i\\omega_1 t}$)")
plt.plot(timelist, np.real(conserved_monomial) + np.imag(conserved_monomial),
         label="Monomial: Re($z_1e^{-i\\omega_1 t}$) + Im($z_1e^{-i\\omega_1 t}$)")
plt.plot(timelist, logc1234, label="Cross-ratio: ln($c_{1234}$)")
plt.plot(timelist, logc2345, label="Cross-ratio: ln($c_{2345}$)")
plt.plot(timelist, S2_2logc2345, label="Symmetry-generated: $\\mathcal{S}_2(2\\ln(c_{2345}))$")
# plt.plot(timelist, S2_alpha0, label="Symmetry-generated: $\\mathcal{S}_2((2/\\sigma_1)\\ln(c_{2345}))$")
plt.plot(timelist, S3_2logc2345, label="Symmetry-generated: $\\mathcal{S}_3(2\\ln(c_{2345}))$")
plt.xlabel("Time $t$")
plt.xticks([0, 5, 10, 15, 20])  
plt.legend(title="Constants of motion", frameon=True, facecolor='white', edgecolor='0.7',
           framealpha=1, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=fontsize_legend)
plt.show()
