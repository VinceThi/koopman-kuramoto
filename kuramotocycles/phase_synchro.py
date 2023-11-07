import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation, colormaps
from dynamics.constants_of_motion import cross_ratio_theta
# from plots.config_rcparams import *


# TODO : Tracer les trajectoires individuelles
# TODO : Observer le comportement avec des petites perturbations



#TEMP
dark_grey = '#404040'                       # RGB: 48, 48, 48     dark grey
first_community_color = "#4c72b0"   # "#2171b5" # RGB: 33, 113, 181   blue
second_community_color = "#e66816"  # "#f16913" # RGB: 241, 105, 19   orange
third_community_color = "#238b45"           # RGB: 35, 139, 69    green
fourth_community_color = "#6a51a3"          # RGB: 106, 81, 163   purple


n_iter = 1000
n = 12
n_parts = 3    # number of clusters
thetas_init_clusters = np.random.rand(n_parts) - np.pi
omegas_clusters = np.random.rand(n_parts)
alpha = 0
a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
print(f"matrice de connectivité: {a}")

clusters = [[0], [1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]]

wrong_clusters = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11]]

# construct initial conditions and natural frequencies for all oscillators
thetas_init = []
omegas = []
for i, cluster in enumerate(clusters):
    for _ in cluster:
        thetas_init.append(thetas_init_clusters[i])
        omegas.append(omegas_clusters[i])

# OPTIONAL : Use the wrong clusters for the initial conditions
thetas_init = []
for i, cluster in enumerate(wrong_clusters):
    for _ in cluster:
        thetas_init.append(thetas_init_clusters[i])

# OPTIONAL : Generate individual random initial conditions instead
# thetas_init = np.random.rand(n) - np.pi

print("conditions initiales :", thetas_init)
print("fréquences naturelles :", omegas)

# def k(t):
#     return 0.1 * t

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
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection="polar")

ax.get_xaxis().set_visible(False)
ax.set_ylim(0, 1.5)
ax.set_rticks([0.5, 1])
ax.set_xticks([])

cm = colormaps.get_cmap('coolwarm')
scat = ax.scatter(thetas_init, [1 for _ in range(n)], marker="o", c=omegas, cmap=cm, linewidth=1, edgecolors=dark_grey)
line1, = ax.plot([], [], marker="o", color=dark_grey)
text_k = ax.text(1.8, 2, '')

lines = [line1]

def animate(i):
    data = np.array([[thetas_mod[i, j], 1] for j in range(n)])
    scat.set_offsets(data)
    text_k.set_text(f"k={valeurs_k[i]:.3f}")
    lines[0].set_data([0, params_ordre[i][1]], [0, params_ordre[i][0]])

    return scat, text_k, lines,

anim = animation.FuncAnimation(fig, animate, interval=20, frames=n_iter, repeat=False)

mp4writer = animation.FFMpegWriter(fps=30)
anim.save('/Users/benja/Desktop/phase_sync_wrongclust.mp4', writer=mp4writer)

plt.show()
