# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45
from plots.kuramoto_animation import animate_kuramoto_on_circle
from matplotlib.animation import FuncAnimation

""" Partition parameters"""
sizes_monomial = np.array([1], dtype=int)
sizes_crossratio = np.array([4], dtype=int)
size_nonintegrable = np.array([], dtype=int)
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
q = len(sizes)  # Number of parts
N = np.sum(sizes, dtype=int)
m = len(sizes_monomial)
c = len(sizes_crossratio)

""" Get weight matrix """
random_exponents = np.array([1])  # np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
probabilities_monomial = np.array([0])
probabilities_crossratio = np.array([[1, 0.5, 0.5, 0.5, 0.5]])  # np.random.rand(c, N)
probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
weights_monomial = np.array([1])  # np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
weights_crossratio = np.array([1, 1, 1, 1, 1])  # np.random.normal(1, 1, (len(sizes_crossratio), N))
weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

W = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                         probabilities=probabilities_dict, weights=weights_dict)
coupling = 1

""" Get phase-lag matrix """
probabilities_monomial2 = [1, 1, 1, 1]
probabilities_crossratio2 = np.ones((c, N))    # np.zeros((c, N))  # np.random.rand(c, N)
probabilities_nonintegrable2 = []
probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
phaselags_monomial = np.random.normal(0, 0.1, (sum(sizes_monomial), sum(sizes_monomial))) # np.zeros((sum(sizes_monomial), sum(sizes_monomial)))
phaselags_crossratio = np.random.normal(0, 0.1, (len(sizes_crossratio), N)) # np.zeros((len(sizes_crossratio), N))
phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
alpha = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                probabilities=probabilities_dict2, phaselags=phaselags_dict)
cal_A = calA(coupling, weights_crossratio, phaselags_crossratio)


""" Get natural frequencies """
omega = np.concatenate([np.array([5*np.random.rand()]),
                         2*np.random.rand()*np.ones((sizes_crossratio[0],))])  # np.array([2, 1, 1, 1, 1])  #
# random_gaussian_frequencies_pintegrable(c, sizes, cal_A, 1, 1)      #

""" Integration parameters """
t0, t1, dt = 0, 40, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))
theta0 = np.random.uniform(0, 2*np.pi, N)
print("init cond = ", theta0)
print("omega = ", omega)
print("W = ", W)
print("alpha = ", alpha)

""" Integrate """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))



""" Illustration of the results"""
interval = 10

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Animate into ax1
ani_ax = animate_kuramoto_on_circle(theta, sizes, interval=interval, ax=ax1)

# Plot time series into ax2
j = 0
for nu in range(len(sizes)):
    for i in range(sizes[nu]):
        ax2.plot(timelist, theta[:, j+i] % (2*np.pi), color=deep[nu], linewidth=1)
    j += sizes[nu]

# Add vertical time bar
vline = ax2.axvline(timelist[0], color='k', linestyle='--', linewidth=1)

# Modify the animation to update vline as well
T = len(theta)
def update_combined(frame):
    # Update ax1 (circle)
    x = np.cos(theta[frame])
    y = np.sin(theta[frame])
    ani_ax._func(frame)  # manually call update from original animation

    # Update vertical bar in ax2
    vline.set_xdata([timelist[frame]])
    return vline,

# Create new animation that updates both ax1 and ax2
ani = FuncAnimation(fig, update_combined, frames=T, interval=interval, blit=False)

ax2.set_ylabel("Phases")
ax2.set_xlabel("Time $t$")
ax2.set_xlim(timelist[0], timelist[-1])
ax2.set_ylim(0, 2*np.pi)

plt.tight_layout()
plt.show()
