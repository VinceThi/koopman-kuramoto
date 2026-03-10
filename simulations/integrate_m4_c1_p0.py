# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
# import matplotlib
# matplotlib.use("Agg")
# from plots.kuramoto_animation import animate_kuramoto_on_circle
from plots.config_rcparams import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from dynamics.dynamics import kuramoto
from dynamics.integrate import integrate_dopri45
from dynamics.watanabe_strogatz import ws_transformation, Z_dot, phi_dot


""" Partition parameters"""
sizes_monomial = np.array([2, 2, 1, 1], dtype=int)
sizes_crossratio = np.array([44], dtype=int)
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

W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                            probabilities=probabilities_dict, weights=weights_dict)
coupling = 0.3

""" Get phase-lag matrix """
probabilities_monomial2 = [1, 1, 1, 1]
probabilities_crossratio2 = np.random.rand(c, N)
probabilities_nonintegrable2 = []
probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
phaselags_monomial = np.random.normal(0, 0.4, (sum(sizes_monomial), sum(sizes_monomial)))
phaselags_crossratio = np.random.normal(0, 0.4, (len(sizes_crossratio), N))
phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
alpha, chi = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                     probabilities=probabilities_dict2, phaselags=phaselags_dict)
cal_A = calA(coupling, weights_crossratio, phaselags_crossratio)


""" Get natural frequencies """
omega = random_gaussian_frequencies_pintegrable(m, c, sizes, cal_A, 0.1, 4)


""" Integration parameters """
t0, t1, dt = 0, 20, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))
theta0 = np.random.uniform(0, 2*np.pi, N)
print("init cond = ", theta0)


""" Integrate """
args_dynamics = (W, coupling, omega, alpha)
theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto, theta0, *args_dynamics))


""" Integrate reduced dynamics """
args_reduced_dynamics = (W, coupling, omega, alpha)
def partially_integrated_equations_m4c1(t, state, w, coupling, omega):
    Z, phi = state
    F_bar = coupling/(2*1j)*np.sum(ws_transformation(Z, phi, w))
    G = omega
    F = np.conjugate(F_bar)
    return np.array([Z_dot(Z, F, G, F_bar), phi_dot(Z, F, G, F_bar)])


""" Illustration of the result """
plt.figure(figsize=(6, 6))
plt.subplot(411)
plt.plot(timelist, theta[:, :2] % (2*np.pi), color=deep[0])
plt.title("Phases $\\theta_1(t), \\theta_2(t)$ (2-source)")

plt.subplot(412)
plt.plot(timelist, theta[:, 2:4] % (2*np.pi), color=deep[1])
plt.title("Phases $\\theta_3(t), \\theta_4(t)$ (2-source)")

plt.subplot(413)
plt.plot(timelist, theta[:, 4] % (2*np.pi), color=deep[2])
plt.plot(timelist, theta[:, 5] % (2*np.pi), color=deep[3])
plt.title("Phases $\\theta_5(t), \\theta_6(t)$ (sources)")

plt.subplot(414)
plt.plot(timelist, theta[:, 6:] % (2*np.pi), color=deep[4])
# plt.plot(timelist, theta_ws, color=deep[1], linestyle="--")
plt.title("Phases $\\theta_7(t), ..., \\theta_N(t)$")
plt.xlabel("Time $t$")
plt.show()

# path = "/Users/vincentthibeault/Documents/PythonProjects/koopman-kuramoto/simulations/animations/kuramoto_anim.mp4"
# animate_kuramoto_on_circle(theta, sizes, interval=30, save_path=path, ax=None)