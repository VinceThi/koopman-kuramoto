import numpy as np
import json
from scipy.linalg import svdvals
from plots.config_rcparams import *

path = "C:/Users/thivi/Documents/GitHub/koopman-kuramoto/symbolic/symmetries/determining_matrices/"
with open(path+"ReImDetMatrix_N4_d1_pm1_2.json", "r") as file:
    json_data = json.load(file)
json_array = np.array(json_data)  # Shape: (rows, cols, 2)

# Combine the real and imaginary parts into the determining matrix
calD = json_array[..., 0] + 1j*json_array[..., 1]

U, S, Vh = np.linalg.svd(calD, full_matrices=True)

plt.matshow(np.real(np.conjugate(Vh.T)), aspect="auto")
plt.show()

print(np.real(np.conjugate(Vh.T)[:, -5:-1]))

v1 = np.real(np.conjugate(Vh.T)[:, -1])
v2 = np.real(np.conjugate(Vh.T)[:, -2])
v3 = np.real(np.conjugate(Vh.T)[:, -3])
v4 = np.real(np.conjugate(Vh.T)[:, -4])


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
