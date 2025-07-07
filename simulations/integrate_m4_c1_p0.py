# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45


""" Partition parameters"""
sizes_monomial = [2, 2, 1, 1]
sizes_crossratio = [50]
size_nonintegrable = [0]
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
q = len(sizes)  # Number of parts
N = sum(sizes)
m = len(sizes_monomial)
c = len(sizes_crossratio)

""" Get weight matrix """
random_exponents = np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
probabilities_monomial = [1, 1, 1, 1]
probabilities_crossratio = np.random.rand(c, N)  # [[1, 0.5, 0.2, 0.7, 0.2, 0.5], [0.1, 0, 0.2, 0.1, 0.8, 0.1]]
probabilities_nonintegrable = []
weights_monomial = np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
weights_crossratio = np.random.normal(1, 1, (len(sizes_crossratio), N))
weights_nonintegrable = np.random.normal(1, 1, (size_nonintegrable[0], N))

W = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                         probabilities_monomial, probabilities_crossratio, probabilities_nonintegrable,
                         weights_monomial, weights_crossratio, weights_nonintegrable)
coupling = 1

""" Get phase-lag matrix """
probabilities_monomial2 = [1, 1, 1, 1]
probabilities_crossratio2 = np.random.rand(c, N)
probabilities_nonintegrable2 = []
phaselags_monomial = np.random.normal(0, 0.1, (sum(sizes_monomial), sum(sizes_monomial)))
phaselags_crossratio = np.random.normal(0, 0.1, (len(sizes_crossratio), N))
phaselags_nonintegrable = np.random.normal(0, 0.1, (size_nonintegrable[0], N))
alpha = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                probabilities_monomial2, probabilities_crossratio2, probabilities_nonintegrable2,
                                phaselags_monomial, phaselags_crossratio, phaselags_nonintegrable)
cal_A = calA(coupling, weights_crossratio, phaselags_crossratio)


""" Get natural frequencies """
omega = random_gaussian_frequencies_pintegrable(c, sizes, cal_A, 1, 1)


""" Integration parameters """
t0, t1, dt = 0, 100, 0.001
timelist = np.linspace(t0, t1, int(t1 / dt))
theta0 = np.random.uniform(0, 2*np.pi, N)
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
# plt.subplot(211)
for i in range(N):
    plt.plot(timelist, theta[:, 0] % (2*np.pi), color=deep[0], linewidth=2)
plt.ylabel("Phases")
plt.xlabel("Time $t$")
# plt.xticks([0, 5, 10, 15, 20])
plt.legend(frameon=True, facecolor='white', edgecolor='0.7',
           framealpha=1, loc='center left', bbox_to_anchor=(1.02, 0.5),
           fontsize=fontsize_legend)
plt.show()
