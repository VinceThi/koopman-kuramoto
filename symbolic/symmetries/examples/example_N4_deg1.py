import numpy as np
import json
from plots.config_rcparams import *

""" Import determining matrix generated with Mathematica """
path = "C:/Users/thivi/Documents/GitHub/koopman-kuramoto/symbolic/symmetries/determining_matrices/"
with open(path+"ReImDetMatrix_N4_d1_pm1_2_source.json", "r") as file:
    json_data = json.load(file)
json_array = np.array(json_data)  # Shape: (rows, cols, 2)


""" Combine the real and imaginary parts into the determining matrix """
calD = json_array[..., 0] + 1j*json_array[..., 1]
# plt.matshow(np.real(calD), aspect="auto")
# plt.show()

""" SVD to extract singular vectors related to zero singular values """
U, S, Vh = np.linalg.svd(calD, full_matrices=True)
# plt.matshow(np.real(np.conjugate(Vh.T)), aspect="auto")
# plt.show()
subV = np.real(np.conjugate(Vh.T)[:, -5:-1])
v1 = np.conjugate(Vh.T)[:, -4]
v2 = np.conjugate(Vh.T)[:, -3]
v3 = np.conjugate(Vh.T)[:, -2]
v4 = np.conjugate(Vh.T)[:, -1]


""" Known symmetries """
L0_singvec = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

calK_singvec = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1j,
                         0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1j, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

""" Ensure that the vectors are indeed singular vectors """
for vec in [v1, v2, v3, v4, L0_singvec.T, np.real(calK_singvec.T), np.imag(calK_singvec.T)]:
    print(np.allclose(calD@vec, np.array(np.zeros(len(calD[:, 0]))), rtol=1e-14, atol=1e-14))
# print(np.allclose(calD@v2, np.array(np.zeros(len(calD[:, 0]))), rtol=1e-14, atol=1e-14))
# print(np.allclose(calD@v3, np.array(np.zeros(len(calD[:, 0]))), rtol=1e-14, atol=1e-14))
# print(np.allclose(calD@v4, np.array(np.zeros(len(calD[:, 0]))), rtol=1e-14, atol=1e-14))
# print(np.allclose(calD@L0_singvec.T, np.array(np.zeros(len(calD[:, 0]))), rtol=1e-14, atol=1e-14))
# print(np.allclose(np.real(calD@calK_singvec.T), np.array(np.zeros(len(calD[:, 0]))), rtol=1e-14, atol=1e-14))
# print(np.allclose(np.imag(calD@calK_singvec.T), np.array(np.zeros(len(calD[:, 0]))), rtol=1e-14, atol=1e-14))


print(np.round(np.real(v1), 5), np.round(np.real(v2), 5), np.round(np.real(v3), 5), np.round(np.real(v4), 5))

# print(L0_singvec.T@calK_singvec) # => they are linearly independent
qL = L0_singvec@subV
qK = calK_singvec@subV
print(np.round(qL, 8), np.round(qK, 8))
# print(np.round(np.real(v1), 5))
print(calK_singvec)
print(np.round(np.real(v1) - 0.01388*L0_singvec[0], 5))


plt.scatter(np.arange(1, len(S) + 1, 1), S)
plt.ylabel("Singular values")
plt.xlabel("Index")
plt.show()

