# -*- coding: utf-8 -*-
# @authors: Vincent Thibeault, Benjamin Claveau, Patrick Desrosiers, Antoine Allard

from pathlib import Path
from dynamics.constants_of_motion import *
import matplotlib.pyplot as plt

plot_weight_matrix = 1

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "graphs" / "adjacency_matrices"
B = np.array(np.load(path/"fly_larva.npy"))  # For advogato: [4017, 4020, 4021, 3512]
N = len(B[:, 0])


if plot_weight_matrix:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.matshow(B, aspect="auto")
    """ To look at a motif admitting a cross-ratio """
    # clique = [257, 966, 842, 1165, 782, 1806, 2131, 1560, 1498, 350, 862, 869, 1383, 2280, 487, 235, 750, 47, 1520, 1521, 1456, 2094, 311, 1784, 375, 1855]
    # ax.matshow(B[np.ix_(clique, clique)], aspect="auto")
    """ To look at a motif admitting a two-vertex monomial """
    # print(np.where(B[:, 130:132])[0])
    # ax.matshow(B[:, 130:132], aspect="auto")
    plt.show()


""" Compute similarity matrix """
S = similarity_matrix_cross_ratio(B)

# """ For pdumerilii_neuronal """
#print("S[1249, 1650] =", S[1249, 1650])
#print("S[1330, 1650] =", S[1330, 1650])
#print("S[1529, 1650] =", S[1529, 1650])
#
#print("S[1249, 1649] =", S[1249, 1649])
#print("S[1330, 1649] =", S[1330, 1649])
#print("S[1529, 1649] =", S[1529, 1649])
#
#print("S[1650, 1649] =", S[1529, 1649])
#
#
# # Check symmetry and transitivity directly
# mask = np.isclose(S, 1+0j, atol=1e-10)
#
# # For each row that has 4 ones, list which other rows it matches
# for i in range(len(S)):
#     others = np.where(mask[i])[0]
#     if len(others) == 4:
#         print(f"Row {i} matches {others}")
#     if len(others) == 5:
#         print(f"Row {i} matches {others}")
#
# rows = [1249, 1330, 1529, 1649]
# tol = 1e-10

# for i in rows:
#     for j in rows:
#         val = S[i, j]
#         ok = np.isclose(val, 1+0j, atol=tol)
#         print(f"S[{i},{j}] = {val: .3e}  {'✓' if ok else '✗'}")
# 
# def neighbors_at_one(S, atol=1e-10):
#     mask = np.isclose(S, 1.0+0j, atol=atol)
#     return [np.where(mask[i])[0] for i in range(S.shape[0])]
# 
# # # Example: inspect your four rows
# rows = [1249, 1330, 1529, 1649]
# nbrs = neighbors_at_one(S, atol=1e-10)
# for i in rows:
#     print(i, "->", nbrs[i])

""" Count motifs related to constants of motion"""
nb_single_sources = count_single_sources(B)
nb_source_pairs = count_source_pairs(B)
nb_conserved_cross_ratios = count_cross_ratio_motifs(B, S)

""" Metrics """
print(f"N = {N}",
      # f"\nrank(W) = {np.linalg.matrix_rank(W)}",
      # f"\nnb_zero_rows = {nb_zero_rows}",
      f"\nnb one-vertex sources = {nb_single_sources}",
      f"\nnb two-vertex sources = {nb_source_pairs}",
      f"\nnb conserved cross-ratios = {nb_conserved_cross_ratios}")


# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.matshow(S, aspect="auto")
# plt.show()

