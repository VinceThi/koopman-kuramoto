import numpy as np
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz import ws_equations_kuramoto, ws_transformation
from dynamics.ws_initial_conditions import get_watanabe_strogatz_initial_conditions
from plots.config_rcparams import *


""" STEP 1: REDUCED SYSTEM """

plot_trajectories = True

""" Parameters """
t0, t1, dt = 0, 20, 0.005
timelist = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
N = 500
W = np.ones((N, N))
omega = 1
coupling = 1/N
np.random.seed(499)
theta0 = np.random.uniform(0, 2*np.pi, N)

# """ Integrate complete dynamics """
# args_dynamics = (W, coupling, omega, alpha)
# theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics)) % (2*np.pi)

""" WS transform and integration """
Z0, phi0, w = get_watanabe_strogatz_initial_conditions(theta0, nb_guess=20)
args_ws = (w, coupling, omega)
solution = np.array(integrate_dopri45(t0, t1, dt, ws_equations_kuramoto, np.array([Z0, phi0]), *args_ws))
Z, phi = solution[:, 0], solution[:, 1]
theta_ws = []
for i in range(len(timelist)):
    theta_ws.append(np.angle(ws_transformation(Z[i], phi[i], w)))
theta_ws = np.array(theta_ws)
theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)

# if plot_trajectories:
    # plt.figure(figsize=(6, 6))
    # plt.plot(timelist, theta[:, 0], color=deep[0], label="original system")
    # plt.plot(timelist, theta[:, 1:], color=deep[0])
    # plt.plot(timelist, theta_ws[:, 0], color=deep[1], linestyle="--", label="watanabe-strogatz")
    # plt.plot(timelist, theta_ws[:, 1:], color=deep[1], linestyle="--")
    # plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
    # plt.xlabel("Time $t$")
    # plt.legend()
    # plt.show()

# print(np.max(np.abs(theta - theta_ws)))

# assert np.all(np.abs(theta - theta_ws) < 1e-6)


""" STEP 2: PERTURBATION """

np.random.seed(10)
n_links_to_remove = [2, 20, 50, 100, 300]
order_params = []
for n in n_links_to_remove:
    links_to_remove = np.random.randint(0, N, (2, n))
    W_pert = W.copy()
    for index in links_to_remove:
        W_pert[index] = 0


    """ STEP 3: INTEGRATE PERTURBED SYSTEM """

    args_dynamics = (W_pert, coupling, omega, alpha)
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics)) % (2*np.pi)


    """ STEP 4: COMPUTE ORDER PARAM AND SHOW RESULTS """

    order_param = np.abs(1/N * np.sum(np.exp(1j * theta), axis=1))
    order_params.append(order_param)

fig, axs = plt.subplots(nrows=1, ncols=len(order_params), figsize=(30, 6))
for i, ax in enumerate(axs):
    ax.set_title(f'{n_links_to_remove[i]} links removed', fontsize=15)
    ax.plot(timelist, np.abs(Z), color=deep[0], label='reduced (no pert)')
    ax.plot(timelist, order_params[i], color=deep[1], label='complete (pert)')
    ax.set_ylabel("Order parameter $R$", fontsize=15)
    ax.set_xlabel("Time $t$", fontsize=15)
    ax.legend()
plt.suptitle('N=500, identical oscillators', fontsize=15)
plt.savefig('/Users/benja/Desktop/n_links_removed_identical_osc.png')
plt.show()
