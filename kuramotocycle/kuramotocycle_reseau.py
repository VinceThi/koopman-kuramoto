import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation, colormaps
from dynamics.constants_of_motion import cross_ratio_theta

n_iter = 1000
n = 5
thetas_init = np.random.rand(n) % (2*np.pi) - np.pi
# thetas_init[1:] = np.random.rand(1) % (2*np.pi) - np.pi + np.random.rand(4) * 0.01    # same initial positions with small perturbation
omegas = np.array([2, 1, 1, 1, 1])
alpha = 0
# a = np.ones((n, n))
# a = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#               [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#               [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#               ])
a = np.array([[0, 1, 1, 1, 1],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0]])
print(f"matrice de connectivité: {a}")

# def k(t):
#     return 0.01 * t

def k(t):
    return 2

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
    params_ordre.append(param_ordre(thetas[-1] % (2*np.pi) - np.pi))
    if integrator.status == 'finished':
        break
thetas = np.array(thetas) % (2*np.pi) - np.pi
r = [param[0] for param in params_ordre]

# plt.figure()
# for i in range(n):
#     plt.plot(temps, thetas[:, i])
# plt.show()


### ANIMATION ###

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(221, projection="polar")
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(224)
ax4 = fig.add_subplot(223)
ax.get_xaxis().set_visible(False)
ax.set_ylim(0, 1.5)
ax.set_rticks([0, 0.5, 1])
ax.set_xticks([])
ax2.plot(np.linspace(0, 100, n_iter), [0 for i in range(n_iter)], "black", lw=0.2)
cm = colormaps.get_cmap('copper')
scat = ax.scatter(thetas_init, [1 for _ in range(n)], c=omegas, cmap=cm)
line, = ax.plot([], [], marker="o", color="black")
text_k = ax.text(1.8, 2, '')

c_1234 = cross_ratio_theta(thetas[:, 1], thetas[:, 2], thetas[:, 3], thetas[:, 4])
c_0123 = cross_ratio_theta(thetas[:, 0], thetas[:, 1], thetas[:, 2], thetas[:, 3])

# R vs k
# ax2.set_xlim(0, max(valeurs_k))
# ax2.set_ylim(0, 1)
# ax2.set_xlabel("Constante de couplage $K$")
# ax2.set_ylabel("Paramètre d'ordre $R$")
# #ax2.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
# ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
# line2, = ax2.plot([], [])

# theta_a, theta_b, theta_c, theta_d
ax2.set_xlim(0, temps[-1])
ax2.set_ylim(-np.pi, np.pi)
ax2.set_xlabel("$t$")


line2, = ax2.plot([], [], ".", markersize=1, label="$\delta_{12}$")
line3, = ax2.plot([], [], ".", markersize=1, label="$\delta_{23}$")
line4, = ax2.plot([], [], ".", markersize=1, label="$\delta_{34}$")
line5, = ax2.plot([], [], ".", markersize=1, label="$\delta_{41}$")

ax2.legend(loc="upper right")

ax3.set_xlim(0, temps[-1])
ax3.set_ylim(-np.pi, np.pi)
ax3.set_xlabel("$t$")

line6, = ax3.plot([], [], linewidth=0.5, label="$θ_1$")
line7, = ax3.plot([], [], linewidth=0.5, label="$θ_2$")
line8, = ax3.plot([], [], linewidth=0.5, label="$θ_3$")
line9, = ax3.plot([], [], linewidth=0.5, label="$θ_4$")
ax3.legend(loc="upper right")


ax4.set_xlim(0, temps[-1])
ax4.set_ylim(-10, 10)
ax4.set_xlabel("$t$")

line10, = ax4.plot([], [], label="$c_{1234}$")
line11, = ax4.plot([], [], label="$c_{0123}$")
ax4.legend(loc="upper right")


lines = [line, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11]

def animate(i):
    data = np.array([[thetas[i, j], 1] for j in range(n)])
    scat.set_offsets(data)
    text_k.set_text(f"k={valeurs_k[i]:.3f}")
    lines[0].set_data([0, params_ordre[i][1]], [0, params_ordre[i][0]])
    # lines[1].set_data(valeurs_k[0:i + 1], r[0:i + 1])
    lines[1].set_data(temps[0:i+1], thetas[0:i+1, 1] - thetas[0:i+1, 2])
    lines[2].set_data(temps[0:i+1], thetas[0:i+1, 2] - thetas[0:i+1, 3])
    lines[3].set_data(temps[0:i+1], thetas[0:i+1, 3] - thetas[0:i+1, 4])
    lines[4].set_data(temps[0:i+1], thetas[0:i+1, 4] - thetas[0:i+1, 1])
    lines[5].set_data(temps[0:i+1], thetas[0:i+1, 1])
    lines[6].set_data(temps[0:i+1], thetas[0:i+1, 2])
    lines[7].set_data(temps[0:i+1], thetas[0:i+1, 3])
    lines[8].set_data(temps[0:i+1], thetas[0:i+1, 4])
    lines[9].set_data(temps[0:i+1], c_1234[0:i+1])
    lines[10].set_data(temps[0:i+1], c_0123[0:i+1])
    return scat, text_k, lines,

anim = animation.FuncAnimation(fig, animate, interval=5, frames=1000, repeat=False)

plt.show()
