import numpy as np
from numpy.linalg import eig
from scipy.linalg import svdvals
from plots.config_rcparams import *

B = np.array([
    [-8j, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [1, -6j, 0, 0, 1, 0, 0, 0, 0, 0],
    [-3, 0, -6j, 0, 1, 0, 0, 0, 0, 0],
    [0, -3, 1, -4j, 0, -1, 1, 0, 0, 0],
    [0, 0, -2, 0, -4j, 0, 0, 0, 0, 0],
    [0, 0, 0, -2, 1, -2j, 0, -2, 1, 0],
    [0, 0, 0, 0, -1, 0, -2j, 0, -1, 0],
    [0, 0, 0, 0, 0, -1, 1, 0, 0, -3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 2j],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

calD = np.array([
    [-8j, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [1, -6j, 0, 0, 1, 0, 0, 0, 0, 0],
    [-3, 0, -6j, 0, 1, 0, 0, 0, 0, 0],
    [0, -3, 1, -4j, 0, -1, 1, 0, 0, 0],
    [0, 0, -2, 0, -4j, 0, 0, 0, 0, 0],
    [0, 0, 0, -2, 1, -2j, 0, -2, 1, 0],
    [0, 0, 0, 0, -1, 0, -2j, 0, -1, 0],
    [0, 0, 0, 0, 0, -1, 1, 0, 0, -3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 2j]])

vepD0 = np.array([0, 0, 0, -3j, 0, -8, 4, 7j, -8j, 4])


calN = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

print(calD@vepD0, calN@vepD0)
# print(svdvals(calD), svdvals(calN))


# A = np.array([[-4*1j, 0,    1,   0,  0,  0,   0,    0],
#               [1,    -2*1j, 0,  -1,  1,  0,   0,    0],
#               [-2,    0,  -2*1j, 0,  0,  0,   0,    0],
#               [0,    -2,    1,   0,  0,  -2,   1,   0],
#               [0,    0,    -1,   0,  0,  0,   -1,   0],
#               [0,    0,     0,   -1, 1,  2*1j, 0,  -3],
#               [0,    0,     0,   0,  0,  0,   2*1j, 0],
#               [0,    0,     0,   0,  0,  0,    1,  4*1j]])
#
# vep0 = np.array([0, 0, 0, 1, 1, 0, 0, 0])      # L_0 generator
# vep1 = np.array([0, -1, 0, 2*1j, 0, 1, 0, 0])  # Koopman generator of the second oscillator
# vep2 = np.array([0, 1.59004109e-01+1.25268691e-01j, 0,
#                  7.51981217e-01, 5.01443834e-01+3.18008219e-01j,
#                  -1.59004109e-01-1.25268691e-01j, 0, 0])  # I think it's not ok, the matrix is non diagonalisable
# vas = svdvals(A)
# # print(vas)
#
# eigvals, eigvecs = eig(A)
#
# # print(eigvals[-3], eigvecs[:, -3])
#
# real_parts = [z.real for z in eigvals]
# imag_parts = [z.imag for z in eigvals]
#
# plt.figure(figsize=(8, 4))
# plt.subplot(121)
# plt.scatter(real_parts, imag_parts)
# ax = plt.gca()
# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
# # plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Real axis
# # plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Imaginary axis
# # plt.xlabel('Real')
# # plt.ylabel('Imaginary')
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# plt.text(xlim[1], 0, "Real", fontsize=12, ha='left', va='center')  # Real axis label
# plt.text(0, ylim[1], "Imaginary", fontsize=12, ha='center', va='bottom')  # Imaginary axis label
# plt.axis('equal')  # Equal scaling
#
# plt.subplot(122)
# plt.scatter(np.arange(1, len(A[0]) + 1, 1), vas)
# plt.ylabel("Singular values")
# plt.xlabel("Index")
# plt.show()
#