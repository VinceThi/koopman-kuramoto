# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import pytest
from graphs.generate_integrability_partitioned_weight_matrix import *

def test_calA():
    """ Partition parameters"""
    sizes_monomial = np.array([1], dtype=int)
    sizes_crossratio = np.array([4, 5], dtype=int)
    size_nonintegrable = np.array([], dtype=int)
    sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
    q = len(sizes)  # Number of parts
    N = np.sum(sizes, dtype=int)
    m = len(sizes_monomial)
    c = len(sizes_crossratio)

    """ Get weight matrix """
    random_exponents = np.array([1])  # np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
    probabilities_monomial = np.array([0])
    probabilities_crossratio = np.array([[1, 0.8, 0],
                                         [1, 0., 0.8]])
    probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio}
    weights_monomial = np.array([1])  # np.random.normal(1, 1, (sum(sizes_monomial), sum(sizes_monomial)))
    weights_crossratio = np.random.normal(1, 1, (len(sizes_crossratio), N))
    weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio}

    W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                                probabilities=probabilities_dict, weights=weights_dict)
    coupling = 1

    """ Get phase-lag matrix """
    probabilities_monomial2 = np.array([0])
    probabilities_crossratio2 = np.array([[1, 0.9, 0],
                                          [1, 0., 0.7]])
    probabilities_nonintegrable2 = []
    probabilities_dict2 = {"monomial": probabilities_monomial2, "crossratio": probabilities_crossratio2}
    phaselags_monomial = np.random.normal(0, 0.1, (sum(sizes_monomial), sum(sizes_monomial)))  # np.zeros((sum(sizes_monomial), sum(sizes_monomial)))
    phaselags_crossratio = np.random.normal(0, 0.1, (len(sizes_crossratio), N))  # np.zeros((len(sizes_crossratio), N))
    phaselags_dict = {"monomial": phaselags_monomial, "crossratio": phaselags_crossratio}
    alpha, chi = random_phase_lag_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable,
                                         probabilities=probabilities_dict2, phaselags=phaselags_dict)

    """ calA """
    # Unit test to verify cal_A
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = (W[1, 0], W[5, 0], W[2, 1], W[1, 2], W[1, 3], W[1, 4],
                                                   W[6, 5], W[5, 6], W[5, 7], W[5, 8], W[5, 9])
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = (alpha[1, 0], alpha[5, 0], alpha[2, 1], alpha[1, 2], alpha[1, 3],
                                                   alpha[1, 4], alpha[6, 5], alpha[5, 6], alpha[5, 7], alpha[5, 8],
                                                   alpha[5, 9])
    cal_Averif = coupling/2*np.array([[w0*np.exp(-1j*a0), w2*np.exp(-1j*a2), w3*np.exp(-1j*a3),
                                       w4*np.exp(-1j*a4), w5*np.exp(-1j*a5), 0., 0., 0., 0., 0.],
                                       [w1*np.exp(-1j*a1),  0., 0., 0., 0., w6*np.exp(-1j*a6), w7*np.exp(-1j*a7),
                                       w8*np.exp(-1j*a8), w9*np.exp(-1j*a9), w10*np.exp(-1j*a10)]])
    cal_A = calA(coupling, C, chi)
    return np.all(np.abs(cal_A - cal_Averif) < 1e-6)

# test_calA()
if __name__ == "__main__":
    pytest.main()