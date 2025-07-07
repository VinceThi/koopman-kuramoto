# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
# from plots.config_rcparams import *


""" See Table 1 in the paper: Partially integrable random graph of Kuramoto oscillators """

# --------------------------------- Monomial part ----------------------------------------------------------------------
def monomial_exponent_matrix(sizes_monomial, random_exponents):
    """

    :param sizes_monomials: a list with the size d_tau of each part of vertices admitting monomial eigenfunctions
    :param random_exponents: -->positive<-- matrix of size N x m
                             with the desired exponents (i.e., it must not contain zeros)
    :return: U := (mu_1  ... mu_m) is a sum_tau d_tau x m array where mu_tau is a sum_tau d_tau x 1 array of
             the tau-th monomial exponent with non-zero real elements. It is a block diagonal matrix with blocks
              of sizes d_tau times 1. Note that it differs from the paper where U is of size N x m
    """
    blocks = [np.ones((d_tau, 1)) for d_tau in sizes_monomial]
    U_binary = block_diag(*blocks)
    return U_binary*random_exponents


def diagonal_exponent_matrix(sizes_monomial, random_exponents):
    """
    :param sizes_monomial: a list with the size d_tau of each part of vertices admitting monomial eigenfunctions
    :param random_exponents: -->positive<-- matrix of size N x m
                             with the desired exponents (i.e., it must not contain zeros)
    :return: D: the diagonal matrix diag(mu_{1,1}, ..., mu_{1,d_1}, mu_{2,d_1+1}, ..., mu_{2,d_1+d_2}, ..., mu_md_m)
    """
    U = monomial_exponent_matrix(sizes_monomial, random_exponents)
    return np.diag(np.sum(U, axis=1))


def symmetric_concatenated_random_matrix(N, sizes_monomial, probabilities_list, weight_matrix):
    """
    :param N: number of oscillators/vertices
    :param sizes_monomial: a list with the size d_tau of each part of vertices admitting monomial eigenfunctions
    :param probabilities_list: list of size len(sizes_monomials) with the connection probabilities between the vertices
           within each monomial part.
    :param weight_matrix: weight matrix of size sum_tau d_tau x sum_tau d_tau of a connected graph (in order to have
           one monomial for each part)
    :return: sum_tau d_tau x N random block symmetric matrix concatenated with a zero block for the connections within
             the monomial parts
    """
    blocks = [(np.random.rand(d_tau, d_tau) < p_tau).astype(int)
              for d_tau, p_tau in zip(sizes_monomial, probabilities_list)]
    print(np.shape(block_diag(*blocks)), np.shape(weight_matrix))
    B = block_diag(*blocks)*weight_matrix
    symB = B + B.T
    return np.concatenate([symB, np.zeros((np.sum(sizes_monomial), N - np.sum(sizes_monomial)))], axis=1)


def skew_symmetric_concatenated_random_matrix(N, sizes_monomial, probabilities_list, weight_matrix):
    """
    :param N: number of oscillators/vertices
    :param sizes_monomial: a list with the size d_tau of each part of vertices admitting monomial eigenfunctions
    :param probabilities_list: list of size len(sizes_monomials) with the connection probabilities between the vertices
           within each monomial part.
    :param weight_matrix: weight matrix of size sum_tau d_tau times sum_tau d_tau with weights within ]-pi/2, pi/2[
    :return: random block skew-symmetric matrix concatenated with a zero block for the phase lags of the monomial parts
    """
    if np.any(np.abs(weight_matrix) >= np.pi/2):
        raise ValueError("The phase-lags between the oscillators must be within ]-pi/2, pi/2[")
    blocks = [(np.random.rand(d_tau, d_tau) < p_tau).astype(int)
              for d_tau, p_tau in zip(sizes_monomial, probabilities_list)]
    beta = block_diag(*blocks)
    skewsym_beta = (beta - beta.T)*weight_matrix
    return np.concatenate([skewsym_beta, np.zeros((np.sum(sizes_monomial), N - np.sum(sizes_monomial)))], axis=1)



# --------------------------------- Cross-ratio part -------------------------------------------------------------------
def membership_crossratio_matrix(sizes_crossratio):
    """ The size of the motifs admitting conserved cross-ratios must be greater than or equal to 4. """
    if not np.all(np.array(sizes_crossratio) >= 4*np.ones(len(sizes_crossratio))):
        raise ValueError("The size of the motifs admitting conserved cross-ratios must be greater than or equal to 4.")
    blocks = [np.ones((n_gamma, 1)) for n_gamma in sizes_crossratio]
    return block_diag(*blocks)


def crossratio_weight_matrix(probabilities_matrix, sizes, weight_matrix):
    """

    :param probabilities_matrix: size c x (m + c + 1), it contains the probabilities of connections between the motifs
           admitting conserved cross-ratios and the monomial, themselves and the non-integrable part
    :param sizes = [d1, ... , dm, n1, ... , nc, p], the size of each part of the partition, n_gamma >= 4 for all gamma
                     monomials,   cross-ratios, non integrable
    :param weight_matrix:
    :return: c x N matrix equal to C^T in the paper
    """
    c = len(probabilities_matrix)  # Number of parts/motif admitting conserved cross-ratios
    q = len(sizes)  # Number of parts in the partition
    for gamma in range(c):
        row_blocks = []
        for nu in range(q):
            row_blocks.append((np.random.rand(1, sizes[nu]) < probabilities_matrix[gamma][nu]).astype(int))
        if not gamma:
            block_matrix = np.block(row_blocks)
        else:
            row_blocks = np.concatenate(row_blocks, axis=1)
            block_matrix = np.concatenate([block_matrix,
                                           row_blocks], axis=0)
    return block_matrix*weight_matrix


def crossratio_phaselag_matrix(probabilities_matrix, sizes, weight_matrix):
    """

    :param probabilities_matrix: size c x (m + c + 1), it contains the probabilities of having phase lags between the
           oscillator's motifs admitting conserved cross-ratios and the monomial, themselves and the non-integrable part
    :param sizes = [d1, ... , dm, n1, ... , nc, p], the size of each part of the partition, n_gamma >= 4 for all gamma
                      monomials,  cross-ratios, non integrable
    :param weight_matrix: phase lag matrix of size c times N with weights within ]-pi/2, pi/2[
    :return: c x N matrix equal to chi^T in the paper
    """
    if np.any(np.abs(weight_matrix) >= np.pi / 2):
        raise ValueError("The phase-lags between the oscillators must be within ]-pi/2, pi/2[")
    c = len(probabilities_matrix)  # Number of parts/motif admitting conserved cross-ratios
    q = len(sizes)  # Number of parts in the partition
    for gamma in range(c):
        row_blocks = []
        for nu in range(q):
            row_blocks.append((np.random.rand(1, sizes[nu]) < probabilities_matrix[gamma][nu]).astype(int))
        if not gamma:
            block_matrix = np.block(row_blocks)
        else:
            row_blocks = np.concatenate(row_blocks, axis=1)
            block_matrix = np.concatenate([block_matrix,
                                           row_blocks], axis=0)
    return block_matrix*weight_matrix


def calA(sigma, weight_matrix, phaselag_matrix):
    """
    :param sigma: global coupling strength
    :param weight_matrix: Nxc weight matrix (result from crossratio_weight_matrix)
    :param phaselag_matrix: Nxc phase lag matrix (result from crossratio_phaselag_matrix)

    :return calA from the paper, cxN complex matrix appearing in the partial integration of the cross-ratio parts
    """
    return sigma/2*weight_matrix*np.exp(1j*phaselag_matrix)


# --------------------------------- Non-integrable part ----------------------------------------------------------------
def nonintegrable_weight_matrix(probabilities_list, sizes, weight_matrix):
    p = sizes[-1]  # Number of vertices in the nonintegrable part
    q = len(sizes)  # Number of parts m + c + 1
    row_blocks = []
    for nu in range(q):
        row_blocks.append((np.random.rand(p, sizes[nu]) < probabilities_list[nu]).astype(int))
    return np.block(row_blocks)*weight_matrix



# --------------------------------- Gathering parts --------------------------------------------------------------------
def random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                         probabilities_monomial, probabilities_crossratio, probabilities_nonintegrable,
                         weights_monomial, weights_crossratio, weights_nonintegrable):
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
    N = sum(sizes)
    D = diagonal_exponent_matrix(sizes_monomial, random_exponents)
    S = symmetric_concatenated_random_matrix(N, sizes_monomial, probabilities_monomial, weights_monomial)
    M = membership_crossratio_matrix(sizes_crossratio)
    C = crossratio_weight_matrix(probabilities_crossratio, sizes, weights_crossratio)
    G = nonintegrable_weight_matrix(probabilities_nonintegrable, sizes, weights_nonintegrable)
    W = np.concatenate([np.linalg.inv(D)@S, M@C, G])
    np.fill_diagonal(W, 0)
    return W


def random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                            probabilities_monomial, probabilities_crossratio, probabilities_nonintegrable,
                            phaselags_monomial, phaselags_crossratio, phaselags_nonintegrable):
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
    N = sum(sizes)
    kappa = skew_symmetric_concatenated_random_matrix(N, sizes_monomial, probabilities_monomial, phaselags_monomial)
    M = membership_crossratio_matrix(sizes_crossratio)
    chi = crossratio_weight_matrix(probabilities_crossratio, sizes, phaselags_crossratio)
    g = nonintegrable_weight_matrix(probabilities_nonintegrable, sizes, phaselags_nonintegrable)
    alpha = np.concatenate([kappa, M@chi, g])
    np.fill_diagonal(alpha, 0)
    return alpha


def random_gaussian_frequencies_pintegrable(c, sizes, calA, mean, std):
    q = len(sizes)  # Number of parts m + c + 1
    m = q - c - 1   # Number of monomial parts
    nm = np.sum(sizes[:m])  # Number of nodes in monomial parts
    reference_frequencies = np.random.normal(mean, std, (1, q - m - 1))[0]
    row_blocks = []
    increment = 0
    for nu in range(q):
        if nu < m or nu > q - 2:
            row_blocks.append(np.random.normal(mean, std, (1, sizes[nu])))
        else:
            omega_list = []
            gamma = nu - m
            ell_gamma = nm + increment  # Reference oscillator label with the cross-ratio part
            for i in range(sizes[nu]):
                j = ell_gamma + i
                # print(gamma, len(reference_frequencies[gamma]))
                omega_list.append(reference_frequencies[gamma] + 2*np.imag(calA[gamma, j] - calA[gamma, ell_gamma]))
            row_blocks.append(np.array([omega_list]))
            increment += sizes[nu]
    return np.concatenate(row_blocks, axis=1)[0]  # TODO TO TEST !!!


if __name__ == "__main__":
    sizes_monomial = [1, 5, 10]
    sizes_crossratio = [4, 10]
    size_nonintegrable = [5]
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
    q = len(sizes)  # Number of parts
    N = sum(sizes)
    random_exponents = np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1

    m = len(sizes_monomial)
    c = len(sizes_crossratio)

    probabilities_monomial = [1, 1, 0.9]
    probabilities_crossratio = [[1, 0.5, 0.2, 0.7, 0.2, 0.5],
                                [0.1, 0, 0.2, 0.1, 0.8, 0.1]]
    probabilities_nonintegrable = [0.5, 0.5, 0.5, 0.1, 0.1, 0.7]
    weights_monomial = np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
    weights_crossratio = np.random.normal(1, 1, (len(sizes_crossratio), N))
    weights_nonintegrable = np.random.normal(1, 1, (size_nonintegrable[0], N))

    W = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                             probabilities_monomial, probabilities_crossratio, probabilities_nonintegrable,
                             weights_monomial, weights_crossratio, weights_nonintegrable)

    probabilities_monomial2 = [0.5, 0.5, 0.5]
    probabilities_crossratio2 = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    probabilities_nonintegrable2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    phaselags_monomial = np.random.normal(0, 0.1, (sum(sizes_monomial), sum(sizes_monomial)))
    phaselags_crossratio = np.random.normal(0, 0.1, (len(sizes_crossratio), N))
    phaselags_nonintegrable = np.random.normal(0, 0.1, (size_nonintegrable[0], N))
    alpha = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                    probabilities_monomial2, probabilities_crossratio2, probabilities_nonintegrable2,
                                    phaselags_monomial, phaselags_crossratio, phaselags_nonintegrable)
    cal_A = calA(1, weights_crossratio, phaselags_crossratio)

    print(random_gaussian_frequencies_pintegrable(c, sizes, cal_A, 1, 1))

    plot_eigenvalues = False

    plt.figure(figsize=(9, 4))
    ax1 = plt.subplot(121)
    ax1.set_title("Weight matrix", pad=15)
    im1 = ax1.matshow(W, aspect="auto")
    plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(122)
    ax2.set_title("Phase lag matrix", pad=15)
    im2 = ax2.matshow(alpha, aspect="auto")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

    if plot_eigenvalues:
        # compute eigenvalues
        eigvals = np.linalg.eigvals(W*np.exp(1j*alpha))

        plt.figure(figsize=(5, 5))
        plt.scatter(eigvals.real, eigvals.imag, s=10)

        plt.axhline(0, linewidth=0.7)
        plt.axvline(0, linewidth=0.7)

        # equal aspect ratio → circles look like circles
        plt.gca().set_aspect('equal', 'box')

        plt.xlabel(r'Re$(\lambda)$')
        plt.ylabel(r'Im$(\lambda)$')
        plt.tight_layout()
        plt.show()
