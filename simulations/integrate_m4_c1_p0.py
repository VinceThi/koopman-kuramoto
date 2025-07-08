# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45
from plots.kuramoto_animation import animate_kuramoto_on_circle


""" Partition parameters"""
sizes_monomial = np.array([2, 2, 1, 1], dtype=int)
sizes_crossratio = np.array([6], dtype=int)
size_nonintegrable = np.array([], dtype=int)
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
q = len(sizes)  # Number of parts
N = np.sum(sizes, dtype=int)
m = len(sizes_monomial)
c = len(sizes_crossratio)

""" Get weight matrix """
random_exponents = np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
probabilities_monomial = [1, 1, 1, 1]
probabilities_crossratio = np.random.rand(c, N)
probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
weights_monomial = np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
weights_crossratio = np.random.normal(1, 1, (len(sizes_crossratio), N))
weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

W = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                         probabilities=probabilities_dict, weights=weights_dict)
coupling = 1

""" Get phase-lag matrix """
probabilities_monomial2 = [1, 1, 1, 1]
probabilities_crossratio2 = np.random.rand(c, N)
probabilities_nonintegrable2 = []
probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
phaselags_monomial = np.random.normal(0, 0.1, (sum(sizes_monomial), sum(sizes_monomial)))
phaselags_crossratio = np.random.normal(0, 0.1, (len(sizes_crossratio), N))
phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
alpha = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                probabilities=probabilities_dict2, phaselags=phaselags_dict)
cal_A = calA(coupling, weights_crossratio, phaselags_crossratio)


""" Get natural frequencies """
omega = random_gaussian_frequencies_pintegrable(c, sizes, cal_A, 1, 1)


""" Integration parameters """
t0, t1, dt = 0, 100, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))
theta0 = np.random.uniform(0, 2*np.pi, N)
print("init cond = ", theta0)

""" Integrate """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))

""" Animation of the result """
animate_kuramoto_on_circle(theta, sizes, interval=1, save_path=0)


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
# plt.subplot(211)
j = 0
for nu in range(len(sizes)):
    for i in range(sizes[nu]):
        plt.plot(timelist, theta[:, j+i] % (2*np.pi), color=deep[nu], linewidth=1)
    j += sizes[nu]
plt.ylabel("Phases")
plt.xlabel("Time $t$")
# plt.xticks([0, 5, 10, 15, 20])
# plt.legend(frameon=True, facecolor='white', edgecolor='0.7',
#            framealpha=1, loc='center left', bbox_to_anchor=(1.02, 0.5),
#            fontsize=fontsize_legend)
plt.show()
