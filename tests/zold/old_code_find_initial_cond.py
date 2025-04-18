import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from dynamics.watanabe_strogatz import ws_transformation
# bounds = ([0] + (N+2)*[float('-inf')], [1] + (N+2)*[float('inf')])
# solution = least_squares(objective_function_init_cond, initial_guess, jac=jacobian_matrix_objective_function,
#                          args=(theta0,), method='trf',  bounds=bounds, ftol=tol, xtol=tol, gtol=tol)
# ^ Pour une raison que j'ignore, l'algorithme ne trouve pas de zéro ... même pour N = 10, vraiment étrange

def ws_transformation_real_params(X, Y, phi, psi):
    return (np.exp(1j*(phi + psi)) + X + 1j*Y)/(1 + np.exp(1j*(phi + psi))*(X - 1j*Y))

# return np.concatenate([np.tan((theta0 - x[1]) / 2) - (1 - x[0]) / (1 + x[0]) * np.tan((x[3:] - x[2])),
#                        np.array([np.sum(np.sin(x[3:])), np.sum(np.cos(x[3:])), np.sum(x[3:])])])


print(R0, Theta0, Phi0)  # , w)
print(objective_function_init_cond(np.concatenate([np.array([R0, Theta0, Phi0]), solution.x[3:]]), theta0))

# from dynamics.constants_of_motion import get_independent_cross_ratios_complete_graph
# from dynamics.constants_w import get_w
# def test_coherence_initial_conditions():
#     N = 100
#     theta0 = 2 * np.pi * np.random.random(N)
#     z0 = np.exp(1j * theta0)
#     cross_ratios = get_independent_cross_ratios_complete_graph(z0)
#     w = get_w(cross_ratios, N, nb_iter=500)
#     Z0, phi0 = get_Z0_phi0(theta0, w)
#     assert np.all(np.abs(z0 - ws_transformation(Z0, phi0, w)) < 1e-10)


# bounds = ([0] + (N+2)*[float('-inf')], [1] + (N+2)*[float('inf')])
# solution = least_squares(objective_function_init_cond, initial_guess, args=(theta0,), bounds=bounds)


""" Old code """


# print(R0, Theta0, Phi0)  # , w)
# print(np.tan((theta0 - Theta0)/2) - (1 - R0)/(1 + R0)*np.tan((solution.x[3:] - Phi0)/2))
# print(objective_function_init_cond(np.concatenate([np.array([R0, Theta0, Phi0]), solution.x[3:]]), theta0))

# In objective function
# """ X, Y, phi, psi_1, ..., psi_N = x,  Z = X + i Y, w_j = e^(i psi_j)"""
# return np.concatenate([np.real(-1j*np.log(ws_transformation_real_params(x[0], x[1], x[2], x[3:]))) - theta0,
#                        np.array([np.sum(np.sin(x[3:])), np.sum(np.cos(x[3:])), np.sum(x[3:])])])
# return np.concatenate([np.real(-1j*np.log(ws_transformation_real_params(x[0], x[1], x[2], x[3:]))) - theta0,
#                        np.array([np.sum(np.sin(x[3:])), np.sum(np.cos(x[3:])), x[3]])])
#                                                                              ^ less restrictive constraint ?

# def get_watanabe_strogatz_initial_conditions_and_w(theta0, N, nb_iter=5000):
#     for i in range(nb_iter):
#         X = np.random.uniform(-1, 1)
#         Y = np.random.uniform(-1, 1)
#         phi = 2*np.pi*np.random.random()
#         psi = 2*np.pi*np.random.random(N)
#         initial_guess = np.concatenate([np.array([X, Y, phi]), psi])
#         solution = root(objective_function_init_cond, initial_guess, args=(theta0,), method='hybr')  # , tol=1e-10)
#         if solution.success:
#             break
#
#     if not solution.success:
#         raise ValueError("The optimization did not converge to successful values of X0, Y0, phi0, w.")
#
#     X0, Y0, phi0 = solution.x[0], solution.x[1], solution.x[2]
#     w = np.exp(1j*solution.x[3:])
#     print(X0 + 1j*Y0, phi0, w)
#     return X0 + 1j*Y0, phi0, w
#
# def objective_function_Z_phi(x, init_theta, w):
#     """ X, Y, phi = x ,  Z = X + i Y"""
#     return np.real(-1j*np.log(ws_transformation_real_params(x[0], x[1], x[2], w[:3]))) - init_theta[:3]
#
#
# def get_Z0_phi0(theta0, w, nb_iter=500):
#     for i in range(nb_iter):
#         X = np.random.uniform(-1, 1)
#         Y = np.random.uniform(-1, 1)
#         phi = 2*np.pi*np.random.random()
#         # print(X, Y, phi, objective_function_Z_phi(np.array([X, Y, phi]), theta0, w))
#         solution = root(objective_function_Z_phi, np.array([X, Y, phi]), args=(theta0, w), method='hybr')  #, tol=1e-10)
#         if solution.success:
#             break
#     if not solution.success:
#         raise ValueError("The optimization did not converge to successful values of Z(0) and phi(0).")
#     X0, Y0, phi0 = solution.x
#     return X0 + 1j*Y0, phi0
#
# # from constants_w import *
# # from constants_of_motion import *
# # N = 100
# # theta0 = 2 * np.pi * np.random.random(N)
# # z0 = np.exp(1j * theta0)
# # cross_ratios = get_independent_cross_ratios_complete_graph(z0)
# # w = get_w(cross_ratios, N, nb_iter=500)
# # Z0, phi0 = get_Z0_phi0(theta0, w)
# # print(Z0, phi0)
#
