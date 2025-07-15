# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np

def kuramoto_order_parameter(theta):
    """ theta : shape T,N """
    return np.abs(np.sum(np.exp(1j*theta), axis=1))/len(theta[0])


def network_order_parameter(theta, coupling, W, alpha):
    """
    theta : shape T,N
    W : weight matrix
    alpha: phase-lag matrix
    If alpha = zero matrix and W is a binary undirected matrix, this is r_uni introduced in Schroder et al., Chaos 2017,
    without the time average.
    Note that the order parameter can take negative values in general. But it is strictly positive for every
    -->stable phase locked state<-- and monotonically increasing with coupling under the conditions
    of Theorem 2 in Schroder et al., Chaos 2017 (sum_ j omega_j = 0 and coupling > critical coupling).
    """
    T, N = theta.shape
    total_weight = np.sum(np.abs(W))
    C = coupling*W*np.cos(alpha)
    S = coupling*W*np.sin(alpha)
    order_param = np.zeros(T)
    for t in range(T):
        c = np.cos(theta[t, :]).T  # column vectors
        s = np.sin(theta[t, :]).T  # column vectors
        trace_jacobian = - c.T@C@c + s.T@S@c - s.T@C@s - c.T@S@s
        order_param[t] = -trace_jacobian/(coupling*total_weight)
    return order_param