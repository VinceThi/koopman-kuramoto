import numpy as np
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from dynamics.watanabe_strogatz_graph import *
from dynamics.ws_initial_conditions_graph import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from plots.config_rcparams import *


def test_z_dot_shape():
    coupling = 1
    omegas_z = np.array([[1],
                        [2],
                        [3]])
    z = np.array([[np.exp(3j)],
                  [np.exp(2j)],
                  [np.exp(1j)]])
    z_and_zeta = np.array([[np.exp(3j)],
                           [np.exp(2j)],
                           [np.exp(1j)],
                           [np.exp(4j)],
                           [np.exp(5j)]])
    adj_submatrix = np.array([[0, 3, 0, 0, 2],
                              [1, 0, 1, 1, 0],
                              [6, 0, 0, 1, 0]])
    result = z_dot_g(z, coupling, omegas_z, adj_submatrix, z_and_zeta)

    assert result.shape == (3, 1)


def test_Z_dot_shape():
    omegas_Z = np.array([[1],
                         [2],
                         [3]])
    Z = np.array([[0.2*np.exp(3j)],
                  [0.8*np.exp(2j)],
                  [0.3*np.exp(1j)]])
    p_1 = np.array([[1],
                    [0.3],
                    [3]])
    p_m1 = np.array([[2],
                     [1.8],
                     [1]])
    result = Z_dot_g(Z, omegas_Z, p_1, p_m1)

    assert result.shape == (3, 1)


def test_phi_dot_shape():
    omegas_Z = np.array([[1],
                         [2],
                         [3]])
    Z = np.array([[0.2*np.exp(3j)],
                  [0.8*np.exp(2j)],
                  [0.3*np.exp(1j)]])
    p_m1 = np.array([[2],
                     [1.8],
                     [1]])
    result = phi_dot_g(Z, omegas_Z, p_m1)

    assert result.shape == (3, 1)


def test_ws_equations_graph_identical_osc():
    print("\nBeginning test_ws_equations_graph_identical_osc ...")
    plot_trajectories = True


    # STEP 1 - GENERATE THE GRAPH

    N = 500
    sizes = [N]
    W = np.ones((N, N))

    # STEP 2 - FIND THE WATANABE-STROGATZ INITIAL CONDITIONS

    np.random.seed(61)
    theta0 = [2*np.pi*np.random.random(size) for size in sizes]
    z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0]
    print('N', np.sum(sizes))
    Z0, phi0, w = get_ws_initial_conditions_graph(theta0, non_integrable_part=False, nb_guess=20)
    err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0)]
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)

    # STEP 3 - INTEGRATE THE ORIGINAL DYNAMICS

    t0, t1, dt = 0, 10, 0.001
    timelist = np.linspace(t0, t1, int(t1 / dt))
    alpha = 0
    omega = 1
    coupling = 1
    args_dynamics = (W, coupling, omega, alpha)
    all_init_theta = np.concatenate(tuple(map(lambda x: x.tolist(), theta0)))
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, all_init_theta, *args_dynamics)) % (2*np.pi)

    # STEP 4 - INTEGRATE THE REDUCED DYNAMICS

    omegas_Z = np.array([[omega for _ in Z0]]).T
    omegas_z = np.array([[]]).T
    W_intparts = np.array([W[0]])

    args_ws = (w, coupling, omegas_Z, omegas_z, W, W_intparts)
    solution = integrate_dopri45(t0, t1, dt, ws_equations_graph, np.concatenate([Z0, phi0]), *args_ws)
    Z = np.array([state[:len(omegas_Z)] for state in solution])
    phi = np.array([state[len(omegas_Z):2*len(omegas_Z)] for state in solution])
    theta_ws = []
    for i in range(len(timelist)):
        zetas = ws_transformation(Z[i], phi[i], w)[0].tolist()
        theta_ws.append(np.angle(zetas))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)

    # (OPTIONAL) SHOW THE TRAJECTORIES

    if plot_trajectories:
        plt.figure(figsize=(6, 6))
        plt.plot(timelist, theta[:, 0], color=deep[0], label="original system")
        plt.plot(timelist, theta[:, 1:], color=deep[0])
        plt.plot(timelist, theta_ws[:, 0], color=deep[1], linestyle="--", label="watanabe-strogatz")
        plt.plot(timelist, theta_ws[:, 1:], color=deep[1], linestyle="--")
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
        plt.xlabel("Time $t$")
        plt.title('test_ws_equations_graph_identical_osc')
        plt.legend()
        plt.show()


    assert True


def test_ws_equations_graph_smallgraph():
    print("\nBeginning test_ws_equations_graph_identical_osc ...")
    plot_trajectories = False


    # STEP 1 - GENERATE THE GRAPH

    N = 5
    sizes = [1, N-1]
    np.random.seed(50)
    W = np.concatenate((np.random.random((1, N)), np.ones((N-1, N))), axis=0)

    # STEP 2 - FIND THE WATANABE-STROGATZ INITIAL CONDITIONS

    np.random.seed(42)
    theta0 = [2*np.pi*np.random.random(size) for size in sizes]
    z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0]
    print('N', np.sum(sizes))
    print('N non int', sizes[0])
    Z0, phi0, w = get_ws_initial_conditions_graph(theta0, nb_guess=10000)
    err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0[1:])]
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)

    # STEP 3 - INTEGRATE THE ORIGINAL DYNAMICS

    t0, t1, dt = 0, 10, 0.005
    timelist = np.linspace(t0, t1, int(t1 / dt))
    alpha = 0
    omega = 1
    coupling = 1
    args_dynamics = (W, coupling, omega, alpha)
    all_init_theta = np.concatenate(tuple(map(lambda x: x.tolist(), theta0)))
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, all_init_theta, *args_dynamics)) % (2*np.pi)


    # STEP 4 - INTEGRATE THE REDUCED DYNAMICS
    # ws_equations_graph(t, state, w_allparts, omegas_Z, omegas_z, adj_matrix)
    omegas_Z = np.array([[omega for _ in Z0]]).T
    omegas_z = np.array([[omega for _ in z0[0]]]).T
    W_intparts = np.take(W, np.cumsum(sizes[:-1]), axis=0)

    # plt.imshow(W)
    # plt.show()
    # plt.imshow(W_intparts)
    # plt.show()

    args_ws = (w, coupling, omegas_Z, omegas_z, W, W_intparts)
    solution = integrate_dopri45(t0, t1, dt, ws_equations_graph, np.concatenate([Z0, phi0, z0[0]]), *args_ws)
    Z = np.array([state[:len(omegas_Z)] for state in solution])
    phi = np.array([state[len(omegas_Z):2*len(omegas_Z)] for state in solution])
    z_nonint = np.array([state[-len(omegas_z):] for state in solution])
    theta_ws = []
    for i in range(len(timelist)):
        zetas = []
        for mu, w_mu in enumerate(w):
            zetas += ws_transformation(Z[i, mu], phi[i, mu], w_mu).tolist()
        theta_ws.append(np.angle(z_nonint[i].tolist() + zetas))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)

    # (OPTIONAL) SHOW THE TRAJECTORIES

    if plot_trajectories:
        plt.figure(figsize=(6, 6))
        plt.plot(timelist, theta[:, 0], color=deep[0], label="original system")
        plt.plot(timelist, theta[:, 1:], color=deep[0])
        plt.plot(timelist, theta_ws[:, 0], color=deep[1], linestyle="--", label="watanabe-strogatz")
        plt.plot(timelist, theta_ws[:, 1:], color=deep[1], linestyle="--")
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
        plt.xlabel("Time $t$")
        plt.legend()
        plt.show()

    assert True


def test_ws_equations_graph_general_case():
    print("\nBeginning ws_equations_graph_general_case ...")
    plot_trajectories = False
    plot_some_trajectories = False
    plot_Z_modulus = False


    # STEP 1 - GENERATE THE GRAPH

    pq = [[0.8, 0.5, 0.5, 0.5, 0.5],
          [0.1, 0.5, 0.3, 0, 0.1],
          [0.2, 0.01, 0.9, 0.1, 0.3],
          [0.1, 0, 0.2, 0.9, 0.3],
          [0.4, 0.5, 0, 0.3, 0.9]]
    sizes = [13, 50, 33, 20, 15]
    max_nbs_zeros_nonintegrable = sizes[0]*np.array(sizes)
    max_nbs_zeros_integrable = np.tile(np.array(sizes), (len(sizes)-1, 1))
    max_nbs_zeros = np.concatenate([np.array([max_nbs_zeros_nonintegrable]), max_nbs_zeros_integrable])
    proportions_of_zeros = np.array([[0.9, 0.7, 0.7, 0.7, 0.7],
                                     [0.05, 0.05, 0.05, 0.05, 0.05],
                                     [0.05, 0.05, 0.05, 0.05, 0.05],
                                     [0.3, 0.3, 0.3, 0.1, 0.3],
                                     [0.3, 0.3, 0.3, 0.3, 0.1]])
    nbs_zeros = np.around(proportions_of_zeros*max_nbs_zeros)
    nbs_zeros = nbs_zeros.astype(int)
    nbs_zeros = nbs_zeros.tolist()
    means = [[0, 0, 0, 0, 0],
             [0.1, 3, 0.5, 0.3, 0.2],
             [-1, -1, -1, -1, -1],
             [0.1, 0.1, 0.1, 2, 0.1],
             [0.1, 0.1, 0.1, 0.1, 1]]
    stds = [[1, 1, 1, 1, 1],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.05, 0.05, 0.5, 0.05, 0.05],
            [0.05, 0.05, 0.05, 1, 0.03],
            [0.04, 0.05, 0.05, 0.1, 0.6]]

    W = integrability_partitioned_block_weight_matrix(pq, sizes, nbs_zeros, means, stds, self_loops=True)

    # STEP 2 - FIND THE WATANABE-STROGATZ INITIAL CONDITIONS

    np.random.seed(42)
    theta0 = [2*np.pi*np.random.random(size) for size in sizes]
    z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0]
    print('N', np.sum(sizes))
    print('N non int', sizes[0])
    Z0, phi0, w = get_ws_initial_conditions_graph(theta0, nb_guess=10000)
    err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0[1:])]
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)

    # STEP 3 - INTEGRATE THE ORIGINAL DYNAMICS

    t0, t1, dt = 0, 10, 0.005
    timelist = np.linspace(t0, t1, int(t1 / dt))
    alpha = 0
    max_omega = 3
    omegas = []
    for size in sizes:
        omegas += [max_omega * np.random.random()] * size
    coupling = 1
    args_dynamics = (W, coupling, omegas, alpha)
    all_init_theta = np.concatenate(tuple(map(lambda x: x.tolist(), theta0)))
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, all_init_theta, *args_dynamics)) % (2*np.pi)

    # STEP 4 - INTEGRATE THE REDUCED DYNAMICS

    omegas_Z = np.array([np.take(omegas, np.cumsum(sizes[:-1]), axis=0)]).T
    omegas_z = np.array([omegas[:sizes[0]]]).T
    W_intparts = np.take(W, np.cumsum(sizes[:-1]), axis=0)

    # plt.imshow(W)
    # plt.show()
    # plt.imshow(W_intparts)
    # plt.show()

    args_ws = (w, coupling, omegas_Z, omegas_z, W, W_intparts)
    solution = integrate_dopri45(t0, t1, dt, ws_equations_graph, np.concatenate([Z0, phi0, z0[0]]), *args_ws)
    Z = np.array([state[:len(omegas_Z)] for state in solution])
    phi = np.array([state[len(omegas_Z):2*len(omegas_Z)] for state in solution])
    z_nonint = np.array([state[-len(omegas_z):] for state in solution])
    theta_ws = []
    for i in range(len(timelist)):
        zetas = []
        for mu, w_mu in enumerate(w):
            zetas += ws_transformation(Z[i, mu], phi[i, mu], w_mu).tolist()
        theta_ws.append(np.angle(z_nonint[i].tolist() + zetas))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)

    # (OPTIONAL) SHOW THE TRAJECTORIES

    if plot_trajectories:
        plt.figure(figsize=(6, 6))
        plt.plot(timelist, theta[:, 0], color=deep[0], label="original system")
        plt.plot(timelist, theta[:, 1:], color=deep[0])
        plt.plot(timelist, theta_ws[:, 0], color=deep[1], linestyle="--", label="watanabe-strogatz")
        plt.plot(timelist, theta_ws[:, 1:], color=deep[1], linestyle="--")
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
        plt.xlabel("Time $t$")
        plt.legend()
        plt.savefig("/Users/benja/Desktop/WS_graph_gencase_1.png")
        plt.show()

    if plot_some_trajectories:
        indices = np.random.randint(0, theta.shape[1], 9)
        plt.figure(figsize=(6, 6))
        plt.plot(timelist, theta[:, 0], color=deep[0], label="original system")
        plt.plot(timelist, theta[:, indices], color=deep[0])
        plt.plot(timelist, theta_ws[:, 0], color=deep[1], linestyle="--", label="watanabe-strogatz")
        plt.plot(timelist, theta_ws[:, indices], color=deep[1], linestyle="--")
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
        plt.xlabel("Time $t$")
        plt.legend()
        plt.savefig("/Users/benja/Desktop/WS_graph_gencase_1_some_trajs.png")
        plt.show()


    if plot_Z_modulus:
        plt.figure(figsize=(6, 6))
        plt.plot(timelist, np.abs(Z), color=deep[0])
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
        plt.xlabel("Time $t$")
        plt.legend()
        plt.show()


    assert True


def test_ws_equations_graph_smallgraph_pert():
    print("\nBeginning test_ws_equations_graph_identical_osc_pert ...")
    plot_trajectories = False

    # STEP 1 - GENERATE THE GRAPH

    N = 5
    sizes = [1, N-1]
    np.random.seed(50)
    W = np.concatenate((np.random.random((1, N)), np.ones((N-1, N))), axis=0)

    # STEP 2 - FIND THE WATANABE-STROGATZ INITIAL CONDITIONS

    np.random.seed(42)
    theta0 = [2*np.pi*np.random.random(size) for size in sizes]
    z0 = [np.exp(1j*theta0_mu) for theta0_mu in theta0]
    print('N', np.sum(sizes))
    print('N non int', sizes[0])
    Z0, phi0, w = get_ws_initial_conditions_graph(theta0, nb_guess=10000)
    err = [np.abs(z0_mu - ws_transformation(Z0[mu], phi0[mu], w[mu])) for mu, z0_mu in enumerate(z0[1:])]
    print("|z0 - ws_transformation(Z0, phi0, w)| = ", err)

    # STEP 3 - INTEGRATE THE ORIGINAL DYNAMICS

    t0, t1, dt = 0, 10, 0.005
    timelist = np.linspace(t0, t1, int(t1 / dt))
    alpha = 0
    omega = 1
    coupling = 1
    pert = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, -0.1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    W_pert = W + pert
    args_dynamics = (W_pert, coupling, omega, alpha)
    all_init_theta = np.concatenate(tuple(map(lambda x: x.tolist(), theta0)))
    theta = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, all_init_theta, *args_dynamics)) % (2*np.pi)

    # STEP 4 - INTEGRATE THE REDUCED DYNAMICS

    omegas_Z = np.array([[omega for _ in Z0]]).T
    omegas_z = np.array([[omega for _ in z0[0]]]).T
    W_intparts = np.take(W, np.cumsum(sizes[:-1]), axis=0)

    # plt.imshow(W)
    # plt.show()
    # plt.imshow(W_intparts)
    # plt.show()

    args_ws = (w, coupling, omegas_Z, omegas_z, W, W_intparts)
    solution = integrate_dopri45(t0, t1, dt, ws_equations_graph, np.concatenate([Z0, phi0, z0[0]]), *args_ws)
    Z = np.array([state[:len(omegas_Z)] for state in solution])
    phi = np.array([state[len(omegas_Z):2*len(omegas_Z)] for state in solution])
    z_nonint = np.array([state[-len(omegas_z):] for state in solution])
    theta_ws = []
    for i in range(len(timelist)):
        zetas = []
        for mu, w_mu in enumerate(w):
            zetas += ws_transformation(Z[i, mu], phi[i, mu], w_mu).tolist()
        theta_ws.append(np.angle(z_nonint[i].tolist() + zetas))
    theta_ws = np.array(theta_ws)
    theta_ws = np.where(theta_ws < 0, 2*np.pi + theta_ws, theta_ws)

    # (OPTIONAL) SHOW THE TRAJECTORIES

    if plot_trajectories:
        plt.figure(figsize=(6, 6))
        plt.plot(timelist, theta[:, 0], color=deep[0], label="original system")
        plt.plot(timelist, theta[:, 1:], color=deep[0])
        plt.plot(timelist, theta_ws[:, 0], color=deep[1], linestyle="--", label="watanabe-strogatz")
        plt.plot(timelist, theta_ws[:, 1:], color=deep[1], linestyle="--")
        plt.ylabel("Phases $\\theta_1(t), ..., \\theta_N(t)$")
        plt.xlabel("Time $t$")
        plt.legend()
        plt.show()

    assert True





