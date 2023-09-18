import numpy as np
from plots.config_rcparams import *
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.constants_of_motion import cross_ratio_theta
from dynamics.integrate import integrate_dopri45
from correlation_divergence_crossings import corr_divergence_crossings
import time


time1 = time.time()

""" Time parameters """
t0, t1, dt = 0, 5, 0.0001
timelist = np.arange(t0, t1, dt)

""" Graph parameters """
graph_str = "motif"

# example from symposium presentation
W = np.array(
    [[0, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 1, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]])

W = W.T


N = len(W[0])

""" Dynamical parameters """
dynamics_str = "kuramoto_sakaguchi"
omega = np.array([2, 2, 2, 2, 1, 1, 1, 1])
coupling = 0.5
alpha = 0

""" Integration """
x0 = 2*np.pi*np.random.random(N)

args_dynamics = (coupling, omega, alpha)
x = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi,
                               W, x0, *args_dynamics))

time2 = time.time()
print("Time:", time2 - time1)


# crossings when cross-ratios diverge or go to zero
cr1234 = cross_ratio_theta(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
indiv1234 = x[:, :4]
bounds = [10**(-3), 10]
t, diff_crossings = corr_divergence_crossings(cr1234, indiv1234, bounds)
# print("CROSSINGS c_1234")
# print(diff_crossings)
# print(t)

cr5678 = cross_ratio_theta(x[:, 4], x[:, 5], x[:, 6], x[:, 7])
indiv5678 = x[:, 3:]
bounds = [0, 10]
t, diff_crossings = corr_divergence_crossings(cr5678, indiv5678, bounds)
# print("CROSSINGS c_5678")
# print(diff_crossings)
# print(t)

cr3456 = cross_ratio_theta(x[:, 2], x[:, 3], x[:, 4], x[:, 5])
indiv3456 = x[:, 2:6]
bounds = [0, 10]
t_3456, diff_crossings_3456 = corr_divergence_crossings(cr3456, indiv3456, bounds)
t_3456 = np.array(t_3456) * dt
# print("CROSSINGS c_3456")
# print(diff_crossings)
# print(t)

cr1478 = cross_ratio_theta(x[:, 0], x[:, 3], x[:, 6], x[:, 7])
indiv1478 = x[:, [0, 3, 6, 7]]
bounds = [0, 10]
t_1478, diff_crossings_1478 = corr_divergence_crossings(cr1478, indiv1478, bounds)
t_1478 = np.array(t_1478) * dt
print("CROSSINGS c_1478")
print(diff_crossings_1478)
print(t_1478)

# plt.plot(diff_crossings, ".")
# plt.show()


# results
fig, axs = plt.subplots(2)

# INDIVIDUAL TIMESERIES
# for j in range(0, N):
#     plt.plot(timelist, x[:, j] % (2*np.pi), color=first_community_color,
#              linewidth=0.3)

# CONSTANT CROSS-RATIOS
axs[0].plot(timelist, cross_ratio_theta(x[:, 0], x[:, 1], x[:, 2], x[:, 3]),
         label="Cross-ratio $c_{1234}$")
axs[0].plot(timelist, cross_ratio_theta(x[:, 4], x[:, 5], x[:, 6], x[:, 7]),
         label="Cross-ratio $c_{5678}$")

# # NON CONSTANT CROSS-RATIOS
axs[0].plot(timelist, cross_ratio_theta(x[:, 2], x[:, 3], x[:, 4], x[:, 5]),
         label="Cross-ratio $c_{3456}$")
axs[0].plot(timelist, cross_ratio_theta(x[:, 0], x[:, 3], x[:, 6], x[:, 7]),
         label="Cross-ratio $c_{1478}$")


axs[0].legend(loc=1, fontsize=fontsize_legend)

axs[0].set_ylim(-10, 10)
axs[0].set_xlim(-0.25, 5.25)

axs[1].plot(t_3456, diff_crossings_3456, "g.", markersize=1, label="crossing 3456")
axs[1].set_xlim(-0.25, 5.25)

axs[1].plot(t_1478, diff_crossings_1478, "r.", markersize=1, label="crossing 1478")
axs[1].set_xlim(-0.25, 5.25)

# INDIVIDUAL TIMESERIES
for j in range(0, N):
    axs[1].plot(timelist, x[:, j] % (2*np.pi), label=f"{j+1}", linewidth=0.3)
axs[1].legend()
axs[1].set_xlim(-0.25, 5.25)
#
# ylab = plt.ylabel('$\\theta_j$', labelpad=20)
# ylab.set_rotation(0)
# plt.xlabel('Time $t$')
# plt.ylim([-1, 2*np.pi + 1])
plt.tick_params(axis='both', which='major')
plt.tight_layout()
plt.show()
