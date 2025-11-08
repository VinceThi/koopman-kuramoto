import matplotlib.pyplot as plt

from dynamics.constants_of_motion import *
from graphs.generate_integrability_partitioned_weight_matrix import *
from graphs.get_real_connectomes import *
from plots.config_rcparams import *
from dynamics.integrate import integrate_dopri45
from dynamics.dynamics import kuramoto_sakaguchi
from graph_tool.spectral import adjacency
# import pytest


def test_similarity_matrix_crossratio_simplest():

    B = np.array([[0, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 0]])
    S = similarity_matrix_cross_ratio(B)
    expected_S = np.ones((4, 4))
    print(S)

    assert np.all(np.abs(S - expected_S) < 1e-10)


def test_similarity_matrix_crossratio_simplest2():
    """ Example to show that one needs to be careful to count the motifs allowing cross-ratios (summing the rows"""
    B = np.array([[0, 0, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [1, 0, 1, 1, 0, 0]])
    S = similarity_matrix_cross_ratio(B)
    print(np.round(S,3))
    print(np.isclose(S, 1.0 + 0j))

    assert True   # np.all(np.abs(S - expected_S) < 1e-10)


def test_similarity_matrix_crossratio_zero_rows():
    B = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [1, 0, 1, 1, 0, 0]])
    S = similarity_matrix_cross_ratio(B)
    print(np.round(S,3))
    print(np.isclose(S, 1.0 + 0j))

    assert True   # np.all(np.abs(S - expected_S) < 1e-10)


def test_count_cross_ratio_motifs():
    """ Get weight matrix """
    sizes_monomial = np.array([1], dtype=int)
    sizes_crossratio = np.array([4, 4, 5, 10, 50], dtype=int)   # There are thus 58 conserved cross-ratios
    size_nonintegrable = np.array([30], dtype=int)
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
    N = np.sum(sizes, dtype=int)
    random_exponents = np.array([1])  # np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
    probabilities_monomial = np.array([0])
    probabilities_crossratio = np.array([[1, 0.1, 0.08, 0.1, 0, 0.05, 0.01],
                                         [0, 0.2, 0., 0.8, 0.7, 0.3, 0.2],
                                         [0, 0.1, 0.02, 0.1, 0.5, 0.3, 0.03],
                                         [1, 0, 0.2, 0., 0.1, 0.09, 0.1],
                                         [0, 0.8, 0.1, 0.2, 0.5, 0, 0]])
    probabilities_nonintegrable = np.array([0.1, 0.1, 0.3, 0.2, 0, 0.01, 0.09])
    probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio,
                          "nonintegrable": probabilities_nonintegrable}
    weights_monomial = np.array([1])  # np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
    weights_crossratio = np.random.normal(1, 1, (len(sizes_crossratio), N))
    weights_nonintegrable = np.random.normal(1, 1, (size_nonintegrable[0], N))
    weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio, "nonintegrable": weights_nonintegrable}

    W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                                probabilities=probabilities_dict, weights=weights_dict)

    """ Get similarity matrix and count conserved cross ratios"""
    B = W > 0
    S = similarity_matrix_cross_ratio(B)
    nb_cross_ratios_totest = count_cross_ratio_motifs(S)
    expected_answer = 58
    print(nb_cross_ratios_totest)
    assert nb_cross_ratios_totest == expected_answer


def test_ensure_cross_ratio_conservation_realnetwork():
    networkName = "pdumerilii_neuronal"
    W = get_connectome_weight_matrix(networkName)
    W = W - np.diag(np.diag(W))
    B = W > 1e-10
    N = len(B[0])
    #     4-node motif              5-node motif                  893-node motif (source nodes)
    #  [1249, 1649, 1330, 1529], [644, 846, 655, 407, 791], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,...

    """ Time parameters """
    t0, t1, dt = 0, 10, 0.01
    timelist = np.linspace(t0, t1, int(t1 / dt))

    """ Dynamical parameters """
    dynamics_str = "kuramoto_sakaguchi"
    coupling = 2/N
    alpha = 0
    omega = 1

    """ Integration """
    print("Begin integration")
    theta0 = 2*np.pi*np.random.random(N)
    args_dynamics = (B, coupling, omega, alpha)
    x = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))
    print("End integration")

    fig = plt.figure(figsize=(4, 4))
    plt.subplot(111)
    for j in range(0, N, 100):
        plt.plot(timelist, x[:, j] % (2 * np.pi), color=first_community_color,
                 linewidth=0.3)
    # plt.plot(timelist, log_cross_ratio_theta(theta0[1], theta0[2], theta0[3], theta0[4])*np.ones(len(timelist)),
    #          label="Log cross-ratio $log(c_{2345})$")
    # plt.plot(timelist, log_cross_ratio_theta(x[:, 1], x[:, 2], x[:, 3], x[:, 4]))
    plt.plot(timelist, log_cross_ratio_theta(x[:, 0], x[:, 1], x[:, 2], x[:, 3]), label="Empty motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 1], x[:, 2], x[:, 3], x[:, 4]), label="Empty motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 2], x[:, 3], x[:, 4], x[:, 5]), label="Empty motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 3], x[:, 4], x[:, 5], x[:, 6]), label="Empty motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 1249], x[:, 1649], x[:, 1330], x[:, 1529]), label="4-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 644], x[:, 846], x[:, 655], x[:, 407]), label="5-node motif 1")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 846], x[:, 655], x[:, 407], x[:, 791]), label="5-node motif 2")
    # [644, 846, 655, 407, 791], [1249, 1649, 1330, 1529]
    # plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 2], x[:, 4], x[:, 3]))
    ylab = plt.ylabel('$\\theta_j$', labelpad=20)
    ylab.set_rotation(0)
    plt.xlabel('Time $t$')
    # plt.ylim([-1, 2*np.pi + 1])
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.legend(loc=1, fontsize=fontsize_legend)
    plt.show()


def test_ensure_cross_ratio_conservation_realnetwork2():
    g = gt.collection.data["power"]
    W = adjacency(g).toarray()
    B = W - np.diag(np.diag(W))  # Remove diagonal elements
    N = len(B[0])
    # [128, 235, 126, 127], [270, 271, 273, 275], [397, 398, 399, 400], [514, 4463, 4464, 4465, 4466], [588, 620, 598, 599, 600, 601, 602]
    # among others
    # print(np.where(B[598, :] == 1)[0])
    # plt.matshow(B[598:602, 550:650], aspect="auto")
    # plt.show()



    """ Time parameters """
    t0, t1, dt = 0, 10, 0.01
    timelist = np.linspace(t0, t1, int(t1 / dt))

    """ Dynamical parameters """
    dynamics_str = "kuramoto_sakaguchi"
    coupling = 2/N
    alpha = 0
    omega = 1

    """ Integration """
    print("Begin integration")
    theta0 = 2*np.pi*np.random.random(N)
    args_dynamics = (B, coupling, omega, alpha)
    x = np.array(integrate_dopri45(t0, t1, dt, kuramoto_sakaguchi, theta0, *args_dynamics))
    print("End integration")

    fig = plt.figure(figsize=(4, 4))
    plt.subplot(111)
    for j in range(0, N, 100):
        plt.plot(timelist, x[:, j] % (2 * np.pi), color=first_community_color,
                 linewidth=0.3)
    # plt.plot(timelist, log_cross_ratio_theta(theta0[1], theta0[2], theta0[3], theta0[4])*np.ones(len(timelist)),
    #          label="Log cross-ratio $log(c_{2345})$")
    # plt.plot(timelist, log_cross_ratio_theta(x[:, 1], x[:, 2], x[:, 3], x[:, 4]))
    plt.plot(timelist, log_cross_ratio_theta(x[:, 128], x[:, 235], x[:, 126], x[:, 127]), label="In 4-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 270], x[:, 271], x[:, 273], x[:, 275]), label="In 4-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 397], x[:, 398], x[:, 399], x[:, 400]), label="In 4-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 514], x[:, 4463], x[:,4464], x[:, 4465]), label="In 5-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 4463], x[:,4464], x[:, 4465], x[:, 4466]), label="In 5-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 588], x[:, 620], x[:, 598], x[:, 599]), label="In 7-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 620], x[:, 598], x[:, 599], x[:, 600]), label="In 7-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 598], x[:, 599], x[:, 600], x[:, 601]), label="In 7-node motif")
    plt.plot(timelist, log_cross_ratio_theta(x[:, 599], x[:, 600], x[:, 601], x[:, 602]), label="In 7-node motif")
    # [644, 846, 655, 407, 791], [1249, 1649, 1330, 1529]
    # plt.plot(timelist, cross_ratio_theta(x[:, 1], x[:, 2], x[:, 4], x[:, 3]))
    ylab = plt.ylabel('$\\theta_j$', labelpad=20)
    ylab.set_rotation(0)
    plt.xlabel('Time $t$')
    # plt.ylim([-1, 2*np.pi + 1])
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.legend(loc=1, fontsize=fontsize_legend)
    plt.show()



def test_count_cross_ratio_motifs_realnetwork():
    networkName = "pdumerilii_neuronal"
    W = get_connectome_weight_matrix(networkName)
    W = W - np.diag(np.diag(W))
    B = W > 1e-10
    S = similarity_matrix_cross_ratio(B)
    # plt.matshow(W > 1e-9)
    # plt.show()
    nb_cross_ratios_totest = count_cross_ratio_motifs(S)
    print(nb_cross_ratios_totest)
    assert nb_cross_ratios_totest == 890 + 3

# test_similarity_matrix_crossratio_simplest()
# test_similarity_matrix_crossratio_simplest2()
# test_similarity_matrix_crossratio_zero_rows()
# test_count_cross_ratio_motifs()
# test_ensure_cross_ratio_conservation_realnetwork()
test_ensure_cross_ratio_conservation_realnetwork2()
# test_count_cross_ratio_motifs_realnetwork()
# if __name__ == "__main__":
#     pytest.main()
