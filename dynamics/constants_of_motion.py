# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def cross_ratio_z(za, zb, zc, zd):
    return (zc - za)*(zd - zb)/((zc - zb)*(zd - za))


def cross_ratio_theta(theta_a, theta_b, theta_c, theta_d):
    return np.sin((theta_c - theta_a)/2)*np.sin((theta_d - theta_b)/2) / \
           np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2)
