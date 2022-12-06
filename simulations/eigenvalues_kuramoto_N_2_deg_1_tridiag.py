# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import cmath
from numpy.linalg import eigvals
from scipy.optimize import leastsq
from plots.config_rcparams import *


def coefficient_matrix_size_2_deg_1(truncation_dimension):
    trd = truncation_dimension
    k = np.arange(-trd, trd, 1)
    d1 = 2*k + 1
    dm1 = -2*k + 1
    return np.diag(d1, k=1) + np.diag(dm1, k=-1)


def symmetrized_coefficient_matrix_size_2_deg_1(truncation_dimension):
    trd = truncation_dimension
    M = coefficient_matrix_size_2_deg_1(trd)
    k = np.arange(-trd, trd, 1)
    arr = (2*k + 1)/(-2*k + 1)
    vec = np.array([cmath.sqrt(x) for x in arr])
    diag = np.concatenate((np.array([1]), vec))
    D = np.diag(diag)
    Dm1 = np.diag(diag**(-1))
    return Dm1@M@D


fontsize_legend = 7
plt.figure(figsize=(9, 3))
ax1 = plt.subplot(131)
plt.title("(a)", fontsize=12)
for td in [2, 10, 15]:  # np.arange(20, 101, 21):

    A = coefficient_matrix_size_2_deg_1(td)
    print(symmetrized_coefficient_matrix_size_2_deg_1(td))
    # print(A)
    eigenvalues = eigvals(A)
    real_part_eig = np.real(eigenvalues)
    imag_part_eig = np.imag(eigenvalues)

    plt.scatter(real_part_eig, imag_part_eig,
                label=f"$d = {td}$", s=8)

plt.ylabel("Im($\\lambda)$")
plt.xlabel("Re($\\lambda$)")
plt.legend(loc="best", fontsize=fontsize_legend)

ax2 = plt.subplot(132)
plt.title("(b)", fontsize=12)
d = 1001
A = coefficient_matrix_size_2_deg_1(d)
vaps = eigvals(A)
vaps_imag = np.imag(eigvals(A))
weights = np.ones_like(vaps_imag) / float(len(vaps_imag))
plt.hist(vaps_imag/np.max(np.abs(vaps_imag)), bins=100,
         color="#064878", edgecolor=None,
         linewidth=1, weights=weights, label=f"$d = {d}$")
plt.ylabel("Spectral density $\\rho(\\lambda)$")
plt.xlabel("Rescaled eigenvalues $\\lambda = iy$")
plt.legend(loc=2, fontsize=fontsize_legend)


# ax3 = plt.subplot(223)
# plt.title("(c)", fontsize=12)
# d_array = np.arange(20, 101, 20)
# for td in d_array:
#
#     A = coefficient_matrix_size_2_deg_1(td)
#     eigenvalues = eigvals(A)
#     imag_part_eig = np.imag(eigenvalues).tolist()
#     sorted_imag_part_eig = np.array(sorted(imag_part_eig, key=abs))
#
#     plt.scatter(np.arange(1, len(sorted_imag_part_eig)+1),
#                 sorted_imag_part_eig,
#                 label=f"$d = {td}$", s=8)
#
# plt.ylabel("Eigenvalues $\\lambda_j = iy_j$")
# plt.xlabel("Index $j$")
# plt.legend(loc=2, fontsize=fontsize_legend)
# plt.ylim([-2.2*max(d_array), 2.2*max(d_array)])


ax4 = plt.subplot(133)
plt.title("(c)", fontsize=12)
d_array = np.array([11, 101, 1001])
# np.concatenate([np.arange(20, 101, 20), np.array([1000])])
for td in d_array:

    A = coefficient_matrix_size_2_deg_1(td)
    eigenvalues = eigvals(A)
    imag_part_eig = np.imag(eigenvalues).tolist()
    sorted_imag_part_eig = np.array(sorted(imag_part_eig, key=abs))
    # sorted_imag_part_eig = np.array(sorted(imag_part_eig))
    if td < max(d_array):
        s = 8
    else:
        s = 1

    rescaled_eigenvalues = sorted_imag_part_eig/td
    rescaled_indices = np.arange(1, len(rescaled_eigenvalues) + 1)/td
    # rescaled_indices=np.linspace(-td//2, td//2, len(rescaled_eigenvalues))/td
    plt.scatter(rescaled_indices, rescaled_eigenvalues,
                label=f"$d = {td}$", s=s)  # , color=deep[i])
    if td == max(d_array):
        # We make a curve fit to the rescaled eigenvalues

        def poly2(x, a1, a2):
            return a1*x + a2*x**2

        def poly3(x, a1, a2, a3):
            return a1*x + a2*x**2 + a3*x**3

        def exp1(x, a1, a2):
            return a1*(np.exp(a2*x) - 1)

        def cost_function(a, x, y, f="exp1"):
            if f == "poly2":
                return y - poly2(x, a[0], a[1])
            elif f == "poly3":
                return y - poly3(x, a[0], a[1], a[2])
            elif f == "exp1":
                return y - exp1(x, a[0], a[1])


        # a_sol = leastsq(func=cost_function,
        #                 x0=np.array([1, 0.05, 0.05, 0.05]),
        #                 args=(rescaled_indices,
        #                       rescaled_eigenvalues,
        #                       "poly3"))[0]
        #
        # plt.plot(rescaled_indices,
        #          poly3(rescaled_indices, a_sol[0], a_sol[1], a_sol[2]),
        #          color=dark_grey, label="Polynomial fit (order 3)")

        # Fit positive eigenvalues
        curve_fit_function = "exp1"
        x0 = np.array([1, 1])
        rescaled_pos_eigenvalues = \
            rescaled_eigenvalues[rescaled_eigenvalues > 0]
        rescaled_pos_indices = \
            np.arange(1, len(rescaled_eigenvalues), 2)/td
        a_sol_pos = leastsq(func=cost_function, x0=x0,
                            args=(rescaled_pos_indices,
                                  rescaled_pos_eigenvalues,
                                  curve_fit_function))[0]

        # Fit negative eigenvalues
        x0 = np.array([-1, 1])
        rescaled_neg_eigenvalues = \
            rescaled_eigenvalues[rescaled_eigenvalues < 0]
        rescaled_neg_indices = \
            np.arange(1, len(rescaled_eigenvalues), 2)/td
        a_sol_neg = leastsq(func=cost_function, x0=x0,
                            args=(rescaled_neg_indices,
                                  rescaled_neg_eigenvalues,
                                  curve_fit_function))[0]

        if curve_fit_function == "poly2":
            plt.plot(rescaled_pos_indices,
                     poly2(rescaled_pos_indices, a_sol_pos[0], a_sol_pos[1]),
                     color=dark_grey, label="Polynomial fit (order 2)")
            plt.plot(rescaled_neg_indices,
                     poly2(rescaled_neg_indices, a_sol_neg[0], a_sol_neg[1]),
                     color=dark_grey)
        elif curve_fit_function == "exp1":
            plt.plot(rescaled_pos_indices,
                     exp1(rescaled_pos_indices, a_sol_pos[0], a_sol_pos[1]),
                     color=dark_grey, label="Exponential fit")
            plt.plot(rescaled_neg_indices,
                     exp1(rescaled_neg_indices, a_sol_neg[0], a_sol_neg[1]),
                     color=dark_grey)

plt.ylabel("Rescaled eigenvalues\n $\\lambda_j/d = iy_j/d$")
plt.xlabel("Rescaled index $j/d$")
plt.legend(loc=2, fontsize=fontsize_legend)
plt.ylim([-5.2, 5.2])

plt.tight_layout()
plt.show()