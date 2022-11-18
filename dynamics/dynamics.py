# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def kuramoto_sakaguchi(t, theta, W, coupling, omega, alpha):
    return omega + coupling*(np.cos(theta+alpha)*(W@np.sin(theta))
                             - np.sin(theta+alpha)*(W@np.cos(theta)))


def complex_kuramoto_sakaguchi(t, z, W, coupling, D, alpha):
    return 1j*D@z + coupling/2*(W@z*np.exp(-1j*alpha)
                                - (z**2)*(W@np.conj(z))*np.exp(1j*alpha))


def theta(t, theta, W, coupling, Iext):
    return 1 - np.cos(theta) + (1 + np.cos(theta)) * \
        (Iext + coupling*(W@(np.ones(len(theta))-np.cos(theta))))


def winfree(t, theta, W, coupling, omega):
    return omega-coupling*np.sin(theta)*(W@(np.ones(len(theta))+np.cos(theta)))
