import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation, colormaps
from dynamics.constants_of_motion import cross_ratio_theta
# from plots.config_rcparams import *



#TEMP
dark_grey = '#404040'                       # RGB: 48, 48, 48     dark grey
first_community_color = "#4c72b0"   # "#2171b5" # RGB: 33, 113, 181   blue
second_community_color = "#e66816"  # "#f16913" # RGB: 241, 105, 19   orange
third_community_color = "#238b45"           # RGB: 35, 139, 69    green
fourth_community_color = "#6a51a3"          # RGB: 106, 81, 163   purple


n_iter = 1000
n = 5
thetas_init = np.random.rand(n) - np.pi
# thetas_init[1:] = np.random.rand(1) % (2*np.pi) - np.pi + np.random.rand(4) * 0.1    # same initial positions with small perturbation
omegas = np.array([2, 1, 1, 1, 1])
alpha = 0
a = np.array([[0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]])
print(f"matrice de connectivité: {a}")

# def k(t):
#     return 0.01 * t

def k(t):
    return 1

def kuramoto(temps, thetas):
    vitesses = []
    for i, theta in enumerate(thetas):
        somme_sins = sum([a[i, j] * np.sin(theta2 - theta - alpha) for j, theta2 in enumerate(thetas)])
        vitesse = omegas[i] + k(temps)/n * somme_sins
        vitesses.append(vitesse)
    return vitesses

def param_ordre(thetas):
    param = np.sum([np.exp(1j * theta) for theta in thetas]) / n
    r = np.abs(param)
    theta = np.angle(param)
    return r, theta

print(f"angles initiaux : {thetas_init}")
print(f"frequences naturelles : {omegas}")

integrator = sp.integrate.RK45(kuramoto, 0, thetas_init, n_iter, 0.1, np.exp(-6))

temps = []
thetas = []
valeurs_k = []
params_ordre = []
for i in range(n_iter):
    integrator.step()
    temps.append(integrator.t)
    thetas.append(integrator.y)
    valeurs_k.append(k(temps[-1]))
    params_ordre.append(param_ordre((thetas[-1] + np.pi) % (2*np.pi) - np.pi))
    if integrator.status == 'finished':
        break
thetas = np.array(thetas)
thetas_mod = (thetas + np.pi) % (2*np.pi) - np.pi
r = [param[0] for param in params_ordre]



### ANIMATION ###

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(221, projection="polar")
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax.get_xaxis().set_visible(False)
ax.set_ylim(0, 1.5)
ax.set_rticks([0.5, 1])
ax.set_xticks([])

ax2.plot(np.linspace(0, 100, n_iter), [0 for i in range(n_iter)], color=dark_grey, lw=0.2)
ax4.plot(np.linspace(0, 100, n_iter), [-2 for i in range(n_iter)], color=dark_grey, lw=0.4)

cm = colormaps.get_cmap('coolwarm')
scat = ax.scatter(thetas_init, [1 for _ in range(n)], marker="o", c=omegas, cmap=cm, linewidth=1, edgecolors=dark_grey)
line, = ax.plot([], [], marker="o", color=dark_grey)
text_k = ax.text(1.8, 2, '')

c_1234 = cross_ratio_theta(thetas[:, 1], thetas[:, 2], thetas[:, 3], thetas[:, 4])
c_0123 = cross_ratio_theta(thetas[:, 0], thetas[:, 1], thetas[:, 2], thetas[:, 3])

delta_12 = (thetas[:, 1] - thetas[:, 2] + np.pi) % (2*np.pi) - np.pi
delta_23 = (thetas[:, 2] - thetas[:, 3] + np.pi) % (2*np.pi) - np.pi
delta_34 = (thetas[:, 3] - thetas[:, 4] + np.pi) % (2*np.pi) - np.pi
delta_41 = (thetas[:, 4] - thetas[:, 1] + np.pi) % (2*np.pi) - np.pi

condition_12 = np.cos((thetas[:, 1] + thetas[:, 2])/2 - thetas[:, 0]) + np.cos((thetas[:, 1] - thetas[:, 3] + delta_23)/2) \
                  + np.cos((thetas[:, 4] - thetas[:, 2] + delta_41)/2)
condition_23 = np.cos((thetas[:, 2] + thetas[:, 3])/2 - thetas[:, 0]) + np.cos((thetas[:, 2] - thetas[:, 4] + delta_34)/2) \
                  + np.cos((thetas[:, 1] - thetas[:, 3] + delta_12)/2)
condition_34 = np.cos((thetas[:, 3] + thetas[:, 4])/2 - thetas[:, 0]) + np.cos((thetas[:, 3] - thetas[:, 1] + delta_41)/2) \
                  + np.cos((thetas[:, 2] - thetas[:, 4] + delta_23)/2)
condition_41 = np.cos((thetas[:, 4] + thetas[:, 1])/2 - thetas[:, 0]) + np.cos((thetas[:, 4] - thetas[:, 2] + delta_12)/2) \
                  + np.cos((thetas[:, 3] - thetas[:, 1] + delta_34)/2)


ax2.set_xlim(0, temps[-1])
ax2.set_ylim(-np.pi, np.pi)
ax2.set_xlabel("$t$")

line2, = ax2.plot([], [], ".", markersize=1, label="$\delta_{12}$", color=first_community_color)
line3, = ax2.plot([], [], ".", markersize=1, label="$\delta_{23}$", color=second_community_color)
line4, = ax2.plot([], [], ".", markersize=1, label="$\delta_{34}$", color=third_community_color)
line5, = ax2.plot([], [], ".", markersize=1, label="$\delta_{41}$", color=fourth_community_color)

ax2.legend(loc="upper right")

ax4.set_xlim(0, temps[-1])
ax4.set_ylim(-np.pi, np.pi)
ax4.set_xlabel("$t$")

line6, = ax4.plot([], [], ".", markersize=1, label="cond 12", color=first_community_color)
line7, = ax4.plot([], [], ".", markersize=1, label="cond 23", color=second_community_color)
line8, = ax4.plot([], [], ".", markersize=1, label="cond 34", color=third_community_color)
line9, = ax4.plot([], [], ".", markersize=1, label="cond 41", color=fourth_community_color)
ax4.legend(loc="upper right")


ax3.set_xlim(0, temps[-1])
ax3.set_ylim(-20, 20)
ax3.set_xlabel("$t$")

line10, = ax3.plot([], [], label="$c_{1234}$", color=first_community_color)
line11, = ax3.plot([], [], label="$c_{0123}$", color=second_community_color)
ax3.legend(loc="upper right")


lines = [line, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11]

def animate(i):
    data = np.array([[thetas_mod[i, j], 1] for j in range(n)])
    scat.set_offsets(data)
    text_k.set_text(f"k={valeurs_k[i]:.3f}")
    lines[0].set_data([0, params_ordre[i][1]], [0, params_ordre[i][0]])
    lines[1].set_data(temps[0:i+1], delta_12[0:i+1])
    lines[2].set_data(temps[0:i+1], delta_23[0:i+1])
    lines[3].set_data(temps[0:i+1], delta_34[0:i+1])
    lines[4].set_data(temps[0:i+1], delta_41[0:i+1])
    lines[5].set_data(temps[0:i+1], condition_12[0:i+1])
    lines[6].set_data(temps[0:i+1], condition_23[0:i+1])
    lines[7].set_data(temps[0:i+1], condition_34[0:i+1])
    lines[8].set_data(temps[0:i+1], condition_41[0:i+1])
    lines[9].set_data(temps[0:i+1], c_1234[0:i+1])
    lines[10].set_data(temps[0:i+1], c_0123[0:i+1])
    return scat, text_k, lines,

anim = animation.FuncAnimation(fig, animate, interval=5, frames=n_iter, repeat=False)

# mp4writer = animation.FFMpegWriter(fps=30)
# anim.save('/Users/benja/Desktop/complete_motif.mp4', writer=mp4writer)

plt.show()
