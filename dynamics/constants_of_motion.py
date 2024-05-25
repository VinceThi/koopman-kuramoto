# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np


def cross_ratio_z(za, zb, zc, zd):
    return (zc - za)*(zd - zb)/((zc - zb)*(zd - za))


def cross_ratio_theta(theta_a, theta_b, theta_c, theta_d):
    # print(theta_c%(2*np.pi), theta_b%(2*np.pi))
    # print(np.sin((theta_c - theta_b)/2))
    # print(np.sin((theta_c - theta_a)/2)*np.sin((theta_d - theta_b)/2))
    # print(np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2))
    return (np.sin((theta_c - theta_a)/2)*np.sin((theta_d - theta_b)/2) /
            np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2))


def log_cross_ratio_theta(theta_a, theta_b, theta_c, theta_d):
    # print(theta_c%(2*np.pi), theta_b%(2*np.pi))
    # print(np.sin((theta_c - theta_b)/2))
    # print(np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2))
    return np.log(np.sin((theta_c - theta_a)/2)**2) \
           + np.log(np.sin((theta_d - theta_b)/2)**2) \
           - np.log(np.sin((theta_c - theta_b)/2)**2) \
           - np.log(np.sin((theta_d - theta_a)/2)**2)


def get_independent_cross_ratios_complete_graph(init_z):
    """ compute the values of the independent cross-ratios from the initial values of the microscopic variables.  """
    cross_ratios = []
    for i, init_z_i in enumerate(init_z[:-3]):
        cross_ratios.append(np.real(cross_ratio_z(init_z_i, init_z[i+1], init_z[i+2], init_z[i+3])))
    return cross_ratios
