# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import cmath
from numpy.linalg import eigvals
from scipy.optimize import leastsq
from plots.config_rcparams import *


def coefficient_matrix_size_2_odd(truncation_dimension):
    trd = truncation_dimension
    i1 = np.arange(-trd, trd, 1)
    im1 = np.arange(-trd + 1, trd + 1, 1)
    d1 = 2*i1 + 3
    dm1 = -2*im1 + 1
    return np.diag(d1, k=1) + np.diag(dm1, k=-1)


def first_similarity_transform_size_2_odd(truncation_dimension):
    N = 2*truncation_dimension + 1
    T = np.eye(N)
    for i in range(1, N+1):
        if i == 1 or i % 2 != 0:
            T[0, -i] = 1
        else:
            T[0, -i] = 0
    return T


def second_similarity_transform_size_2_odd(truncation_dimension):
    N = 2*truncation_dimension
    S = np.eye(N)
    for i in range(1, N):
        if i == 1 or i % 2 != 0:
            S[i-1, i] = -1
        else:
            S[i-1, i] = -1
    return S


trd = 1
R = coefficient_matrix_size_2_odd(trd)
T = first_similarity_transform_size_2_odd(trd)
S = second_similarity_transform_size_2_odd(trd)
Tinv = np.linalg.inv(T)
print(f"O_d =\n {R}")
# print(f"\n\nT =\n {T}", f"\n\nTinv =\n {Tinv}", "\n\n", f"S =\n {S}")
Rsim = T@R@Tinv
M = Rsim[1:, 1:]
print(f"\n\nT O_d T^-1 =\n {Rsim}")
# print(f"\n\nMS =\n{M}n{S}=\n {M@S}\n\nSR =\n {S@(R[1:, 1:])}")
# print(np.linalg.eig(M)[0], np.linalg.eig(R[1:, 1:])[0])
# [[0, 0],
#  [*, M]]


def symmetrized_coefficient_matrix_size_2_odd(truncation_dimension):
    trd = truncation_dimension
    M = coefficient_matrix_size_2_odd(trd)
    diag_list = [1]
    for k in range(-trd+1, trd+1):
        num = (-1)**(k + trd + 1)*(2*trd - 1)
        denom = 2*k + 1
        diag_list.append(cmath.sqrt((num/denom)))
    diagonal = np.array(diag_list)
    D = np.diag(diagonal)
    Dinv = np.diag(diagonal**(-1))
    return Dinv@M@D


def coefficient_matrix_size_2_even(truncation_dimension):
    trd = truncation_dimension
    i1 = np.arange(-trd, trd, 1)
    im1 = np.arange(-trd + 1, trd + 1, 1)
    d1 = 2*i1 + 2
    dm1 = -2*im1 + 2
    return np.diag(d1, k=1) + np.diag(dm1, k=-1)


print(symmetrized_coefficient_matrix_size_2_odd(2))
"""
Symmetrized O_3   (Note: sqrt(3) = 1.73205081..., sqrt(15) = 3.87298335...)                                                        
[[0           -1.73205081j 0              0              0        ]
 [-1.73205081j 0           1              0              0       ]
 [0            1           0              -1.73205081j   0      ]
 [0            0           -1.73205081j   0             3.87298335j]
 [0            0           0.+0.j         0.+3.87298335j  0        ]]

"""
fontsize_legend = 7
plt.figure(figsize=(11, 6))
ax1 = plt.subplot(231)
plt.title("(a)", fontsize=12)
for td in [15, 10, 5]:  # np.arange(20, 101, 21):

    A = coefficient_matrix_size_2_odd(td)
    # print(A)

    eigenvalues = eigvals(A)
    real_part_eig = np.real(eigenvalues)
    imag_part_eig = np.imag(eigenvalues)

    plt.scatter(real_part_eig, imag_part_eig,
                label=f"$d = {td}$", s=8)

    # A_sym = symmetrized_coefficient_matrix_size_2_odd(td)
    # eigenvalues_sym = eigvals(A_sym)
    # real_part_eig_sym = np.real(eigenvalues_sym)
    # imag_part_eig_sym = np.imag(eigenvalues_sym)
    # plt.scatter(real_part_eig_sym, imag_part_eig_sym,
    #             label=f"$d = {td}$", s=2)

# plt.xlim([-1, 1])
plt.ylabel("Im($\\lambda)$")
plt.xlabel("Re($\\lambda$)")
plt.legend(loc="best", fontsize=fontsize_legend)

ax2 = plt.subplot(232)
plt.title("(b)", fontsize=12)
d = 1000
A = coefficient_matrix_size_2_odd(d)
vaps = eigvals(A)
vaps_imag = np.imag(eigvals(A))
weights = np.ones_like(vaps_imag) / float(len(vaps_imag))
plt.hist(vaps_imag/np.max(np.abs(vaps_imag)), bins=100,
         color="#064878", edgecolor=None,
         linewidth=1, weights=weights, label=f"$d = {d}$")
plt.ylabel("Spectral density $\\rho(\\lambda)$")
plt.xlabel("Rescaled eigenvalues $\\lambda = iy/|y_{max}|$")
plt.legend(loc=2, fontsize=fontsize_legend)

ax3 = plt.subplot(233)
plt.title("(c)", fontsize=12)
d_array = np.array([10, 100, 1000])
for td in d_array:

    A = coefficient_matrix_size_2_odd(td)
    eigenvalues = eigvals(A)
    imag_part_eig = np.imag(eigenvalues).tolist()
    sorted_imag_part_eig = np.array(sorted(imag_part_eig, key=abs))
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

        def exp2(x, a1, a2):
            return a1*(a2**x - 1)

        def cost_function(a, x, y, f="exp1"):
            if f == "poly2":
                return y - poly2(x, a[0], a[1])
            elif f == "poly3":
                return y - poly3(x, a[0], a[1], a[2])
            elif f == "exp1":
                return y - exp1(x, a[0], a[1])
            elif f == "exp2":
                return y - exp2(x, a[0], a[1])

        # Fit positive eigenvalues
        # curve_fit_function = "exp1"
        # x0 = np.array([1, 1])
        curve_fit_function = "exp2"
        x0 = np.array([1, np.exp(1)])
        rescaled_pos_eigenvalues = \
            rescaled_eigenvalues[rescaled_eigenvalues >= 0]
        rescaled_pos_indices = \
            np.arange(1, len(rescaled_eigenvalues) + 2, 2)/td
        print(len(rescaled_pos_indices), len(rescaled_pos_eigenvalues))
        a_sol_pos = leastsq(func=cost_function, x0=x0,
                            args=(rescaled_pos_indices,
                                  rescaled_pos_eigenvalues,
                                  curve_fit_function))[0]
        print(a_sol_pos)

        # Fit negative eigenvalues
        # x0 = np.array([-1, 1])
        x0 = np.array([-1, np.exp(1)])
        rescaled_neg_eigenvalues = \
            rescaled_eigenvalues[rescaled_eigenvalues <= 0]
        rescaled_neg_indices = \
            np.arange(1, len(rescaled_eigenvalues) + 2, 2)/td
        a_sol_neg = leastsq(func=cost_function, x0=x0,
                            args=(rescaled_neg_indices,
                                  rescaled_neg_eigenvalues,
                                  curve_fit_function))[0]
        print(a_sol_neg)

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
        elif curve_fit_function == "exp2":
            plt.plot(rescaled_pos_indices,
                     exp2(rescaled_pos_indices, a_sol_pos[0], a_sol_pos[1]),
                     color=dark_grey, label="Exponential fit")
            plt.plot(rescaled_neg_indices,
                     exp2(rescaled_neg_indices, a_sol_neg[0], a_sol_neg[1]),
                     color=dark_grey)


plt.ylabel("Rescaled eigenvalues\n $\\lambda_j/d = iy_j/d$")
plt.xlabel("Rescaled index $j/d$")
plt.legend(loc=2, fontsize=fontsize_legend)
plt.ylim([-5.2, 5.2])


ax4 = plt.subplot(234)
plt.title("(d)", fontsize=12)
for td in [15, 10, 5]:  # np.arange(20, 101, 21):

    A = coefficient_matrix_size_2_even(td)
    # print(A)

    eigenvalues = eigvals(A)
    real_part_eig = np.real(eigenvalues)
    imag_part_eig = np.imag(eigenvalues)

    plt.scatter(real_part_eig, imag_part_eig,
                label=f"$d = {td}$", s=8)
plt.ylabel("Im($\\lambda)$")
plt.xlabel("Re($\\lambda$)")
plt.legend(loc="best", fontsize=fontsize_legend)


ax5 = plt.subplot(235)
plt.title("(e)", fontsize=12)
d = 1000
A = coefficient_matrix_size_2_even(d)
vaps = eigvals(A)
vaps_imag = np.imag(eigvals(A))
weights = np.ones_like(vaps_imag) / float(len(vaps_imag))
plt.hist(vaps_imag/np.max(np.abs(vaps_imag)), bins=100,
         color="#064878", edgecolor=None,
         linewidth=1, weights=weights, label=f"$d = {d}$")
plt.ylabel("Spectral density $\\rho(\\lambda)$")
plt.xlabel("Rescaled eigenvalues $\\lambda = iy/|y_{max}|$")
plt.legend(loc=2, fontsize=fontsize_legend)

ax6 = plt.subplot(236)
plt.title("(f)", fontsize=12)
d_array = np.array([10, 100, 1000])
for td in d_array:

    A = coefficient_matrix_size_2_even(td)
    eigenvalues = eigvals(A)
    imag_part_eig = np.imag(eigenvalues).tolist()
    sorted_imag_part_eig = np.array(sorted(imag_part_eig, key=abs))
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

        def exp2(x, a1, a2):
            return a1*(a2**x - 1)

        def cost_function(a, x, y, f="exp1"):
            if f == "poly2":
                return y - poly2(x, a[0], a[1])
            elif f == "poly3":
                return y - poly3(x, a[0], a[1], a[2])
            elif f == "exp1":
                return y - exp1(x, a[0], a[1])
            elif f == "exp2":
                return y - exp2(x, a[0], a[1])

        # Fit positive eigenvalues
        # curve_fit_function = "exp1"
        # x0 = np.array([1, 1])
        curve_fit_function = "exp2"
        x0 = np.array([1, np.exp(1)])
        rescaled_pos_eigenvalues = \
            rescaled_eigenvalues[rescaled_eigenvalues >= 0]
        rescaled_pos_indices = \
            np.arange(1, len(rescaled_eigenvalues)+2, 2)/td
        a_sol_pos = leastsq(func=cost_function, x0=x0,
                            args=(rescaled_pos_indices,
                                  rescaled_pos_eigenvalues,
                                  curve_fit_function))[0]
        print(a_sol_pos)

        # Fit negative eigenvalues
        # x0 = np.array([-1, 1])
        x0 = np.array([-1, np.exp(1)])
        rescaled_neg_eigenvalues = \
            rescaled_eigenvalues[rescaled_eigenvalues <= 0]
        rescaled_neg_indices = \
            np.arange(1, len(rescaled_eigenvalues) + 2, 2)/td
        a_sol_neg = leastsq(func=cost_function, x0=x0,
                            args=(rescaled_neg_indices,
                                  rescaled_neg_eigenvalues,
                                  curve_fit_function))[0]
        print(a_sol_neg)

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
        elif curve_fit_function == "exp2":
            plt.plot(rescaled_pos_indices,
                     exp2(rescaled_pos_indices, a_sol_pos[0], a_sol_pos[1]),
                     color=dark_grey, label="Exponential fit")
            plt.plot(rescaled_neg_indices,
                     exp2(rescaled_neg_indices, a_sol_neg[0], a_sol_neg[1]),
                     color=dark_grey)


plt.ylabel("Rescaled eigenvalues\n $\\lambda_j/d = iy_j/d$")
plt.xlabel("Rescaled index $j/d$")
plt.legend(loc=2, fontsize=fontsize_legend)
plt.ylim([-5.2, 5.2])


plt.tight_layout()
plt.show()
