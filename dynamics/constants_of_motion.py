# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def cross_ratio_z(za, zb, zc, zd):
    return (zc - za)*(zd - zb)/((zc - zb)*(zd - za))


def cross_ratio_theta(theta_a, theta_b, theta_c, theta_d):
    # print(theta_c%(2*np.pi), theta_b%(2*np.pi))
    # print(np.sin((theta_c - theta_b)/2))
    print(np.sin((theta_c - theta_a)/2)*np.sin((theta_d - theta_b)/2))
    print(np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2))
    return np.sin((theta_c - theta_a)/2)*np.sin((theta_d - theta_b)/2) / \
        (np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2))


def log_cross_ratio_theta(theta_a, theta_b, theta_c, theta_d):
    # print(theta_c%(2*np.pi), theta_b%(2*np.pi))
    # print(np.sin((theta_c - theta_b)/2))
    # print(np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2))
    return np.log(np.sin((theta_c - theta_a)/2)**2) \
           + np.log(np.sin((theta_d - theta_b)/2)**2) \
           - np.log(np.sin((theta_c - theta_b)/2)**2) \
           - np.log(np.sin((theta_d - theta_a)/2)**2)


