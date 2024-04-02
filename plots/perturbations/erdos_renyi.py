import numpy as np
from scipy.integrate import solve_ivp
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz_graph import *
from dynamics.ws_initial_conditions_graph import *
from plots.config_rcparams import *


np.random.seed(80)

""" STEP 1: REDUCED SYSTEM """

plot_trajectories = True

""" Parameters """
t0, t1, dt = 0, 2000, 0.005
time = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
N = 50
p_list = [0.1, 0.08, 0.06, 0.04, 0.02]
thetas = []
order_params = []
order_params_WS = []
sizes = [0, N]
for p in p_list:
    W = np.ones((N, N)) * p
    W_pert = np.random.random((N, N))
    W_pert = (W_pert < p).astype(int)
    pert = W_pert - W
    omega = 1
    coupling = 1/N
    theta0 = [2*np.pi*np.random.random(size) for size in sizes]
    z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0]
    all_init_theta = np.concatenate(tuple(map(lambda x: x.tolist(), theta0)))

    """ WS transform and integration """
    Z0, phi0, w = get_ws_initial_conditions_graph(theta0, nb_guess=20)
    err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0[1:])]
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)

    omegas_Z = np.array([[omega for _ in Z0]]).T
    omegas_z = np.array([[]]).T
    W_intparts = np.array([W[0]])

    print('reduced system integration')
    args_ws = (w, coupling, omegas_Z, omegas_z, W, W_intparts)
    solution = solve_ivp(ws_equations_graph, (t0, t1), np.concatenate([Z0, phi0]), method='DOP853', args=args_ws,
                         t_eval=time, atol=1e-10, rtol=1e-10)
    Z = np.array([state[:len(omegas_Z)] for state in solution.y.T])
    phi = np.array([state[len(omegas_Z):2*len(omegas_Z)] for state in solution.y.T])
    zeta = ws_transformation(Z, phi, w)

    order_params_WS.append(np.abs(1/N * np.sum(zeta, axis=1)))


    """ STEP 3: INTEGRATE PERTURBED SYSTEM """

    print('perturbed system integration')
    args_dynamics = (W_pert, coupling, omega, alpha)
    solution = solve_ivp(kuramoto_sakaguchi, (t0, t1), all_init_theta, method='DOP853', args=args_dynamics,
                     t_eval=time, atol=1e-10, rtol=1e-10)
    print('success', solution.success)
    theta = solution.y.T
    thetas.append(theta)

    """ STEP 4: COMPUTE ORDER PARAM AND SHOW RESULTS """

    order_param = np.abs(1/N * np.sum(np.exp(1j * theta), axis=1))
    order_params.append(order_param)

# PANEL FIGURE
fig, axs = plt.subplots(nrows=1, ncols=len(order_params), figsize=(30, 6))
for i, ax in enumerate(axs):
    ax.set_title(f'p = {p_list[i]}', fontsize=15)
    ax.plot(time, order_params_WS[i], color=deep[0], label='reduced (no pert)')
    ax.plot(time, order_params[i], color=deep[1], label='complete (pert)')
    ax.set_ylabel("Order parameter $R$", fontsize=15)
    ax.set_xlabel("Time $t$", fontsize=15)
    ax.legend()
plt.suptitle('N=50, erdos-renyi', fontsize=15)
plt.savefig('/Users/benja/Desktop/erdos_renyi_50.png')
plt.show()

# SINGLE GRAPH
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), dpi=201)
# ax.plot(time, order_param_WS, color=deep[0], linewidth=1.5, label='reduced (no pert)')
# for i, order_param in enumerate(order_params):
    # ax.plot(time, order_param, '--', color=deep[i+1], label=f'p = {p:.2f}')
# ax.set_ylabel("Order parameter $R$")
# ax.set_xlabel("Time $t$")
# plt.subplots_adjust(right=1.)
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
# plt.suptitle('N=200, MPIN')
# plt.savefig('/Users/benja/Desktop/erdos_renyi_100.png')
# plt.show()

# Show difference between original and perturbed matrices
# plt.figure()
# plt.imshow(W - W_pert)
# plt.colorbar()
# plt.show()

# plot some trajectories
# plt.figure()
# plt.plot(timelist, thetas[0], color='orange')
# plt.show()
