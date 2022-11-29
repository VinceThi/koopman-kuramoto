# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import eigvals
from plots.config_rcparams import *
from tqdm import tqdm


def plot_eigenvalues_histogram_random_matrices(random_matrix_generator,
                                               random_matrix_args,
                                               nb_networks=1000,
                                               nb_bins=1000,
                                               bar_color="#064878",
                                               xlabel="Eigenvalues $\\sigma$",
                                               ylabel="Spectral density"
                                                      " $\\rho(\\sigma)$"):
    plt.figure(figsize=(6, 4))
    eigenvalues = np.array([])
    i = 0
    for k in tqdm(range(0, nb_networks)):
        A = random_matrix_generator(*random_matrix_args)
        eigenvalues_instance = eigvals(A)
        eigenvalues = np.concatenate((eigenvalues, eigenvalues_instance))
        i += 1
    weights = np.ones_like(eigenvalues) / float(len(eigenvalues))
    plt.hist(eigenvalues, bins=nb_bins,
             color=bar_color, edgecolor=None,
             linewidth=1, weights=weights)
    plt.tick_params(axis='both', which='major')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, labelpad=20)
    plt.tight_layout()
    plt.show()
