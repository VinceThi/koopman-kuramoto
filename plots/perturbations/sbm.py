import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz_graph import *
from dynamics.ws_initial_conditions_graph import *
from plots.config_rcparams import *


np.random.seed(61)

""" STEP 1: REDUCED SYSTEM """

plot_trajectories = True

""" Parameters """
t0, t1, dt = 0, 700, 0.001
time = np.linspace(t0, t1, int(t1 / dt))
alpha = 0
N = 150
p_intra = 0.9
p = np.ones((3, 3)) * (1 - p_intra)
np.fill_diagonal(p, p_intra)

# p = [[0.8, 0.8, 0.8],
     # [0.8, 0.8, 0.8],
     # [0.8, 0.8, 0.8],]

sizes_sbm = [int(N/3), int(N/3), N - 2 * int(N/3)]
print(sizes_sbm)
sbm = nx.stochastic_block_model(sizes_sbm, p, directed=True, seed=23)
W_pert = nx.to_numpy_array(sbm)
plt.imshow(W_pert)
plt.colorbar()
plt.show()
sizes = [0] + sizes_sbm
print(sizes)

W = np.zeros((N, N))
cumsum_sizes = np.cumsum(sizes)
for i, partsum_1 in enumerate(cumsum_sizes[1:]):
    for j, partsum_2 in enumerate(cumsum_sizes[1:]):
        W[cumsum_sizes[i]:partsum_1, cumsum_sizes[j]:partsum_2] = p[i][j]
pert = W_pert - W


max_omega = 1
omegas = []
for size in sizes:
    omegas += [max_omega * np.random.random()] * size

coupling = 1/N
theta0 = [2*np.pi*np.random.random(size) for size in sizes]
z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0]
all_init_theta = np.concatenate(tuple(map(lambda x: x.tolist(), theta0)))

thetas = []
order_params = []
order_params_WS = []

""" WS transform and integration """
Z0, phi0, w = get_ws_initial_conditions_graph(theta0, nb_guess=20)
err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0[1:])]
print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)

omegas_z = np.array([[]]).T
omegas_Z = np.array([np.take(omegas, np.cumsum(sizes[:-1]), axis=0)]).T
W_intparts = np.take(W, np.cumsum(sizes[:-1]), axis=0)

print('reduced system integration')
args_ws = (w, coupling, omegas_Z, omegas_z, W, W_intparts)
solution = solve_ivp(ws_equations_graph, (t0, t1), np.concatenate([Z0, phi0]), method='DOP853', args=args_ws,
                     t_eval=time, atol=1e-10, rtol=1e-10)
Z = np.array([state[:len(omegas_Z)] for state in solution.y.T])
phi = np.array([state[len(omegas_Z):2*len(omegas_Z)] for state in solution.y.T])

for mu, Z_mu in enumerate(Z.T):
    zeta_mu = np.array([ws_transformation(Z_mu[i], phi[i, mu], w[mu]) for i, _ in enumerate(time)])
    order_param_mu = 1/sizes[mu+1] * np.abs(np.sum(zeta_mu, axis=1))
    order_params_WS.append(order_param_mu)

""" STEP 3: INTEGRATE PERTURBED SYSTEM """

print('perturbed system integration')
args_dynamics = (W_pert, coupling, omegas, alpha)
solution = solve_ivp(kuramoto_sakaguchi, (t0, t1), all_init_theta, method='DOP853', args=args_dynamics,
                 t_eval=time, atol=1e-10, rtol=1e-10)
print('success', solution.success)
theta = solution.y.T
thetas.append(theta)

""" STEP 4: COMPUTE ORDER PARAM AND SHOW RESULTS """

order_params = []
for mu, partsum in enumerate(cumsum_sizes[1:]):
    order_param = np.abs(1/sizes[mu+1] * np.sum(np.exp(1j * theta[:, cumsum_sizes[mu]:partsum]), axis=1))
    order_params.append(order_param)

# PANEL FIGURE
fig, axs = plt.subplots(nrows=1, ncols=len(order_params), figsize=(24, 6))
for i, ax in enumerate(axs):
    ax.set_title(f'PIB {i+1}', fontsize=15)
    ax.plot(time, order_params_WS[i], color=deep[0], label='reduced (no pert)')
    ax.plot(time, order_params[i], color=deep[1], label='complete (pert)')
    ax.set_ylabel("Order parameter $R$", fontsize=15)
    ax.set_xlabel("Time $t$", fontsize=15)
    ax.legend()
plt.suptitle(f'N=150, SBM (p = {p_intra})', fontsize=15)
plt.savefig(f'/Users/benja/Desktop/sbm_{N}_p{p_intra}.png')
plt.show()

# SINGLE GRAPH
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), dpi=201)
ax.plot(time, order_params_WS[0], color=deep[0], linewidth=1.5, label='reduced (no pert)')
for i, order_param in enumerate(order_params):
    ax.plot(time, order_param, '-', color=deep[i+1], label=f'PIB {i}')
ax.set_ylabel("Order parameter $R$")
ax.set_xlabel("Time $t$")
ax.set_ylim(0.8, 1)
plt.subplots_adjust(right=1.)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.suptitle(f'N={N}, SBM (p = {p_intra})')
plt.savefig('/Users/benja/Desktop/oscillations.png')
plt.show()

# Show difference between original and perturbed matrices
# plt.figure()
# plt.imshow(W - W_pert)
# plt.colorbar()
# plt.show()

# plot some trajectories
# plt.figure()
# plt.plot(timelist, thetas[0], color='orange')
# plt.show()
