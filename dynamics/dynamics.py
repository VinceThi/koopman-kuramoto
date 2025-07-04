# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def kuramoto(t, theta, W, coupling, omega, alpha):
    """ This is the general Kuramoto dynamics defined in Thibeault et al., Kuramoto meets Koopman:..., 2025. """
    C = coupling*W*np.cos(alpha)
    S = coupling*W*np.sin(alpha)
    return omega + np.cos(theta)*(C@np.sin(theta)) - np.sin(theta)*(C@np.cos(theta)) \
        - np.sin(theta)*(S@np.sin(theta)) - np.cos(theta)*(S@np.cos(theta))


def kuramoto_sakaguchi(t, theta, W, coupling, omega, alpha):
    return omega + coupling*(np.cos(theta+alpha)*(W@np.sin(theta)) - np.sin(theta+alpha)*(W@np.cos(theta)))


def ricatti(t, z, theta, current_index, omega, coupling):
    p1 = coupling/2*np.sum(np.exp(1j*theta[current_index, :]))
    return p1 + 1j*omega*z - np.conj(p1)*z**2


def complex_kuramoto_sakaguchi(t, z, W, coupling, D, alpha):
    return 1j*D@z + coupling/2*(W@z*np.exp(-1j*alpha) - (z**2)*(W@np.conj(z))*np.exp(1j*alpha))


def theta(t, theta, W, coupling, Iext):
    return 1 - np.cos(theta) + (1 + np.cos(theta)) * (Iext + coupling*(W@(np.ones(len(theta))-np.cos(theta))))


def winfree(t, theta, W, coupling, omega):
    return omega-coupling*np.sin(theta)*(W@(np.ones(len(theta))+np.cos(theta)))
