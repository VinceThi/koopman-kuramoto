# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from graphs.generate_integrability_partitioned_weight_matrix import monomial_exponent_matrix


def tilde_omega(U, omega):
    if len(omega) != len(U[:, 0]):
        omega = omega[:len(U[:, 0])]
    return U.T @ omega


def coefficients_matrix_a(tilde_omega):
    m = len(tilde_omega)
    e = np.eye(m, m)
    a = np.zeros((m, m-1))
    for eta in range(m - 1):
        a[:, eta] = tilde_omega[eta + 1]*e[:, eta] - tilde_omega[eta]*e[:, eta+1]
    return a


def exponents_conserved_monomials(U, a):
    return U@a


def indices_conserved(sizes_monomial):
    partition = []
    max_indices = []
    i = 0
    j = 0
    for size in sizes_monomial:
        update = np.arange(size) + i
        partition.append(update.tolist())
        i += size
        if j != len(sizes_monomial) - 1:
            max_indices.append(max(update))
        j += 1
    return set(max_indices)


def indices_notconserved(sizes_monomial):
    return set(np.arange(np.sum(sizes_monomial))) - indices_conserved(sizes_monomial)


def permutate_indices_conservation(sizes_monomial):
    i_conserved = indices_conserved(sizes_monomial)
    i_notconserved = indices_notconserved(sizes_monomial)
    P = [i_notconserved, i_conserved]
    return np.array([int(i) for block in P for i in sorted(block)], dtype=int)


def rearranged_exponents_conserved_monomials(sizes_monomial, exponents_conserved_monomials):
    m = len(sizes_monomial)
    hatV = np.zeros((m-1, np.sum(sizes_monomial)))
    permutations = permutate_indices_conservation(sizes_monomial)
    for eta in range(len(exponents_conserved_monomials[0, :])):
        nu_eta = exponents_conserved_monomials[:, eta]
        hatnu_eta = nu_eta[permutations]
        hatV[eta, :] = hatnu_eta
    return hatV


def monomial_coordinate_change_matrix(sizes_monomial, hatV):
    m = len(sizes_monomial)
    sum_dtau = np.sum(sizes_monomial)
    d = sum_dtau - m + 1
    hatI = np.concatenate([np.eye(d, d), np.zeros((d, m - 1))], axis=1)
    return np.concatenate([hatI, hatV])


def kappa_shift(sizes_monomial, V, theta0):
    m = len(sizes_monomial)
    permutations = permutate_indices_conservation(sizes_monomial)
    hattheta0 = theta0[permutations]
    phi_conserved = (V@hattheta0)[-(m-1):]
    Vinv = np.linalg.inv(V)
    return Vinv[-(m-1):, -(m-1):]@phi_conserved


if '__main__' == __name__:
    sizes_monomial = [5, 2, 2, 1, 1]
    calI = indices_conserved(sizes_monomial)
    calM = set(np.arange(np.sum(sizes_monomial)))
    calMI = calM - calI
    print(calMI, calI)
    print(len(calMI), len(calI))


    random_exponents = np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 1
    U = monomial_exponent_matrix(sizes_monomial, random_exponents)
    print("U = ", U)

    omega = np.random.normal(0.2, 1, size=np.sum(sizes_monomial))
    tom = tilde_omega(U, omega)
    print("tilde_omega = ", tom)

    a = coefficients_matrix_a(tom)
    print("a coeffs = ", a)

    nu_expo = exponents_conserved_monomials(U, a)
    print("nu_expo = ", nu_expo)

    hatV = rearranged_exponents_conserved_monomials(sizes_monomial, nu_expo)
    print("hatV = ", hatV)

    V = monomial_coordinate_change_matrix(sizes_monomial, hatV)
    print("V = ", V)

    theta0 = np.random.random(np.sum(sizes_monomial))
    print(kappa_shift(sizes_monomial, V, theta0))
