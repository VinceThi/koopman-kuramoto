# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
import numpy as np
from graphs.adjacency_to_graph import adjacency_matrix_to_graphtool_graph
import graph_tool.all as gt

""" Cross-ratios """
def cross_ratio_z(za, zb, zc, zd):
    return (zc - za)*(zd - zb)/((zc - zb)*(zd - za))


def cross_ratio_theta(theta_a, theta_b, theta_c, theta_d):
    # print(theta_c%(2*np.pi), theta_b%(2*np.pi))
    # print(np.sin((theta_c - theta_b)/2))
    # print(np.sin((theta_c - theta_a)/2)*np.sin((theta_d - theta_b)/2))
    # print(np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2))
    return (np.sin((theta_c - theta_a)/2)*np.sin((theta_d - theta_b)/2) /
            (np.sin((theta_c - theta_b)/2)*np.sin((theta_d - theta_a)/2)))


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


def similarity_matrix_cross_ratio(B):
    """
    B: Binary matrix N times N with N > 3 (otherwise, it is impossible to have conserved cross-ratios)
    return: similarity matrix [complex array (N, N)]

    Example: Consider the binary matrices
    B1 = (0  0  0  1;             B2 = (0  0  1  1;
          1  0  0  1;                   1  0  1  1;
          1  0  0  1;                   1  0  0  1;
          1  0  0  0)                   1  0  1  0)
    These matrices admit the possibility of a conserved cross-ratio and we consider the rows in each respective
    matrices to be similar (they only differ because a diagonal of zeros is going through them). The idea to do such
    thing is to replace the off-diagonal zero elements with the imaginary unit. In this way, two rows are identical when
    the scalar product of two rows divided by N - 2 is equal to one. The diagonal elements are (N-1)/(N-2) and thus,
    subtracting by np.diag(1/(N-2)) provide a diagonal of ones, as desired (a row is similar to itself).

    Important note: If (1) row A and row B have a similarity of 1, row B and row C have a similarity of 1, than it does
    NOT mean that row A and row C have a similarity of 1. In other words, transitivity is not guaranteed.
    Indeed, consider the matrix (0 1 0;  row A
                                 0 0 1;  row B
                                 0 1 0). row C
    Then, on the one hand, row A is obviously similar. On the other hand, row B and row C are considered similar in the
    way we compute similarity. We have to deal with this when establishing how to count the number of motifs
    admitting conserved cross-ratios (count_cross_ratio_motifs).
    """
    B0 = (np.abs(B) > 0.5).astype(np.int8)  # robust to tiny noise; treat as binary
    N = B0.shape[0]
    if N < 4:
        raise ValueError("Need N >= 4 to have the possibility of a conserved cross-ratio.")
    A = B0.astype(complex).copy()
    np.fill_diagonal(A, 0.j)
    off_diag = ~np.eye(N, dtype=bool)
    A[(A == 0) & off_diag] = 1j  # Replace off-diagonal zeros with 1j (A. Allard's nice idea)
    return (A@A.conj().T)/(N - 2) - np.diag(np.ones(N, dtype=complex)/(N - 2))  # The subtraction converts the diagonal elements go to one


def count_cross_ratio_motifs(B, S, atol=1e-8, rtol=1e-10):
    """
    Build an undirected graph where an edge (i,j) exists iff S_ij = 1,
    find connected components that are maximal cliques, and then sum (n-3) over all maximal cliques of size n >= 4.
    """
    # 1) Get binary matrix from the similarity matrix, using where there are ones
    mask = np.isclose(S, 1.0 + 0j, atol=atol, rtol=rtol)
    if not np.allclose(mask, mask.T):
        diff = np.max(np.abs(mask.astype(float) - mask.T.astype(float)))
        raise AssertionError(f"The 0-1 matrix extracted from the similarity matrix is not symmetric (max diff={diff})")

    # 2) Build graph-tool graph
    g = adjacency_matrix_to_graphtool_graph(mask)

    # 3) Count maximal cliques
    nb_cross_ratio_motifs, sizes, motifs_label, max_indeg_motif_list  = 0, [], [], []
    for c in gt.max_cliques(g): # each c is a list of vertex objects
        # Note: this does NOT recompute gt.max_cliques(g) at every loop, it avoids having it in cache
        clique = [int(v) for v in c]
        clique_size = len(clique)
        if clique_size >= 4:
            nb_cross_ratio_motifs += clique_size - 3
            sizes.append(clique_size)
            motifs_label.append(clique)

            Bmotif = B[np.ix_(clique, clique)]
            indeg_motif = np.sum(Bmotif, axis=1).astype(int)
            max_indeg_motif_list.append(int(indeg_motif.max()))

    return nb_cross_ratio_motifs, sizes, motifs_label, max_indeg_motif_list


""" Monomials """
def count_single_sources(W):
    """
    W : any real matrix
    return: number of single-vertex sources in the graph
    """
    # kin = np.sum(W, axis=1)  # One way to do it
    # np.count_nonzero(kin == 0)
    zero_rows = np.all(np.isclose(W, 0, atol=1e-10), axis=1)
    return np.sum(zero_rows)


def count_source_pairs(W):
    """
    W : any real matrix
    return: number of sources pairs, i.e., the number of connected vertex pairs being a source within the whole network
    """
    # Convert to boolean/binary
    B = ~np.isclose(W, 0, atol=1e-10)   # does (W != 0) with a tolerance

    # Mutual (bidirectional) edges
    M = B & B.T  # True where B[i,j]=B[j,i]=1

    # Nodes with in-degree exactly 1 (sum over rows)
    indeg1 = (B.sum(axis=1) == 1)

    # Pairs where mutual edge exists AND both nodes have in-degree 1
    pair_mask = M & indeg1[:, None] & indeg1[None, :]

    # Keep only i<j to avoid double counting
    iu = np.triu(pair_mask, k=1)
    pairs = np.transpose(np.nonzero(iu))
    count = pairs.shape[0]

    return count, pairs
