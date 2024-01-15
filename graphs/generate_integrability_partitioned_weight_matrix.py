# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import matplotlib.pyplot as plt
from graph_tool.all import *


""" Generate block matrix according to the partial integrability partition """


def generate_partially_integrable_block(size, row):
    """
    size: (int) size of the integrable part
    row: (array) contains the weights of the ingoing edges of each vertex in the integrable part
         it will be a repeated row in the weight matrix of the graph

    Returns : an array with the row concatenated one below each other "size" amount of time.
              The shape of the array is (size, len(row)).
    """
    return np.tile(row, (size, 1))


def shuffled_binary_matrix(nb_rows, nb_columns, nb_zeros):
    assert nb_zeros <= nb_rows*nb_columns
    N = nb_rows*nb_columns
    arr = np.array([0]*nb_zeros + [1]*(N - nb_zeros))
    np.random.shuffle(arr)
    return np.reshape(arr, (nb_rows, nb_columns))


def holed_random_gaussian_matrix(nb_rows, nb_columns, nb_zeros, mean, std):
    return shuffled_binary_matrix(nb_rows, nb_columns, nb_zeros)*np.random.normal(mean, std, (nb_rows, nb_columns))


def sliced_random_gaussian_matrix(nb_rows, nb_columns, nb_zeros, mean, std):
    assert nb_zeros <= nb_columns
    slicer = np.array([0]*nb_zeros + [1]*(nb_columns - nb_zeros))
    np.random.shuffle(slicer)
    row = slicer*np.random.normal(mean, std, (1, nb_columns))
    return generate_partially_integrable_block(nb_rows, row)


def integrability_partitioned_block_weight_matrix(pq, sizes, nbs_zeros, means, stds, self_loops=True):
    """ The first element in sizes must be the number of vertices in the non-integrable part. """
    if not np.all(np.array(sizes)[1:] >= 4*np.ones(len(sizes)-1)):
        raise ValueError("The size of the partially integrable parts must be greater than or equal to 4.")
    q = len(sizes)
    for mu in range(q):
        row_blocks = []
        for nu in range(q):
            if not mu:  # if it is the non-integrable part
                row_blocks.append(pq[mu][nu]*holed_random_gaussian_matrix(sizes[mu], sizes[nu], nbs_zeros[mu][nu],
                                                                          means[mu][nu], stds[mu][nu]))
            else:       # else, it is a partially integrable part
                row_blocks.append(pq[mu][nu]*sliced_random_gaussian_matrix(sizes[mu], sizes[nu], nbs_zeros[mu][nu],
                                                                           means[mu][nu], stds[mu][nu]))
        if not mu:
            block_matrix = np.block(row_blocks)
        else:
            row_blocks = np.concatenate(row_blocks, axis=1)
            block_matrix = np.concatenate([block_matrix,
                                           row_blocks], axis=0)
    if not self_loops:
        np.fill_diagonal(block_matrix, 0)
    return block_matrix


if __name__ == "__main__":
    pq = [[0.8, 0.5, 0.5, 0.5, 0.5],
          [0.1, 1, 0.3, 0, 0.1],
          [0.2, 0.01, 0.9, 0.1, 0.3],
          [0.1, 0, 0.2, 0.9, 0.3],
          [0.4, 0.5, 0, 0.3, 0.9]]
    sizes = [38, 4, 58, 150, 250]
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
    print(nbs_zeros)
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

    plt.matshow(W, aspect="auto")
    plt.show()
