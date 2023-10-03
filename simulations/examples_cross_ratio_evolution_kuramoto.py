import numpy as np
from plots.config_rcparams import *
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.constants_of_motion import cross_ratio_theta
from dynamics.integrate import integrate_dopri45
from divergence_and_crossings import mindist_at_divergence
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

# W = np.random.randint(0, 2, size=(8, 8))

print(W)

N = len(W[0])

""" Dynamical parameters """
dynamics_str = "kuramoto_sakaguchi"
omega = np.array([2, 2, 2, 2, 1, 1, 1, 1])
coupling = 1
alpha = 0

""" Integration """
x0 = 2*np.pi*np.random.random(N)

args_dynamics = (coupling, omega, alpha)
x = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi,
                               W, x0, *args_dynamics))

time2 = time.time()
print("Time:", time2 - time1)


# CROSSING WHEN CROSS-RATIOS DIVERGE OR GO TO ZERO

cr1234 = cross_ratio_theta(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
indiv1234 = x[:, :4]
bounds = [0, 10]
t_1234, diff_crossings_1234 = mindist_at_divergence(cr1234, indiv1234, bounds)
t_1234 = np.array(t_1234) * dt

cr5678 = cross_ratio_theta(x[:, 4], x[:, 5], x[:, 6], x[:, 7])
indiv5678 = x[:, 3:]
bounds = [0, 10]
t_5678, diff_crossings_5678 = mindist_at_divergence(cr5678, indiv5678, bounds)
t_5678 = np.array(t_5678) * dt

cr3456 = cross_ratio_theta(x[:, 2], x[:, 3], x[:, 4], x[:, 5])
indiv3456 = x[:, 2:6]
bounds = [0, 10]
t_3456, diff_crossings_3456 = mindist_at_divergence(cr3456, indiv3456, bounds)
t_3456 = np.array(t_3456) * dt

cr1478 = cross_ratio_theta(x[:, 0], x[:, 3], x[:, 6], x[:, 7])
indiv1478 = x[:, [0, 3, 6, 7]]
bounds = [0, 10]
t_1478, diff_crossings_1478 = mindist_at_divergence(cr1478, indiv1478, bounds)
t_1478 = np.array(t_1478) * dt


# results
fig, axs = plt.subplots(2)

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
axs[0].set_xlim(-0.25, 7)

axs[1].plot(t_1234, diff_crossings_1234, ".", markersize=1, label="crossing 3456")
axs[1].set_xlim(-0.25, 7)

axs[1].plot(t_5678, diff_crossings_5678, ".", markersize=1, label="crossing 3456")
axs[1].set_xlim(-0.25, 7)

axs[1].plot(t_3456, diff_crossings_3456, ".", markersize=1, label="crossing 3456")
axs[1].set_xlim(-0.25, 7)

axs[1].plot(t_1478, diff_crossings_1478, ".", markersize=1, label="crossing 1478")
axs[1].set_xlim(-0.25, 7)

# INDIVIDUAL TIMESERIES
for j in range(0, N):
    axs[1].plot(timelist, x[:, j] % (2*np.pi), label=f"{j+1}", linewidth=0.3)
axs[1].legend()
axs[1].set_xlim(-0.25, 7)
#
# ylab = plt.ylabel('$\\theta_j$', labelpad=20)
# ylab.set_rotation(0)
# plt.xlabel('Time $t$')
# plt.ylim([-1, 2*np.pi + 1])
plt.tick_params(axis='both', which='major')
plt.tight_layout()
plt.show()
