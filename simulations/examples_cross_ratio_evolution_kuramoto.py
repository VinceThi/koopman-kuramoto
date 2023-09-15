import numpy as np
from plots.config_rcparams import *
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.constants_of_motion import cross_ratio_theta,\
    log_cross_ratio_theta
from dynamics.integrate import integrate_dopri45, integrate_rk4, integrate_dynamics
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

# W = np.array([[1, 1, 1, 1],
#               [1, 1, 1, 1],
#               [1, 1, 1, 1],
#               [1, 1, 1, 1]])

N = len(W[0])

""" Dynamical parameters """
dynamics_str = "kuramoto_sakaguchi"
omega = np.array([1, 1, 1, 1, 2, 2, 2, 2])
coupling = 1
alpha = 0

""" Integration """
x0 = 2*np.pi*np.random.random(N)

args_dynamics = (coupling, omega, alpha)
x = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi,
                               W, x0, *args_dynamics))

time2 = time.time()
print("Time:", time2 - time1)


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


# INDIVIDUAL TIMESERIES
for j in range(0, 4):
    axs[1].plot(timelist, x[:, j] % (2*np.pi), label=f"{j+1}", linewidth=0.3)
axs[1].legend()

ylab = plt.ylabel('$\\theta_j$', labelpad=20)
ylab.set_rotation(0)
plt.xlabel('Time $t$')
# plt.ylim([-1, 2*np.pi + 1])
plt.tick_params(axis='both', which='major')
plt.show()
