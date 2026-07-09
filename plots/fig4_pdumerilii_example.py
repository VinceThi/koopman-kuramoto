# -*- coding: utf-8 -*-
# @authors: Vincent Thibeault
from pathlib import Path
from dynamics.constants_of_motion import *
from graphs.get_graph_properties import is_symmetric, count_edges_from_binary_matrix
import graph_tool.all as gt


plot_weight_matrix = 0
draw_graph = 0

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "graphs" / "adjacency_matrices"

results = []

B = np.array(np.load(path/f"connectome_pdumerilii_neuronal.npy"))
kin = np.sum(B, axis=1)
kout = np.sum(B, axis=0)

print(np.shape(B))


""" connectome_pdumerilii_neuronal constants of motion """

print("2-source")
print("[1851, 2054]")
print("kin :", kin[1851], kin[2054])
print("kout :", kout[1851], kout[2054])

print("\n4-motif admitting cross-ratio")
print("[1249, 1649, 1330, 1529]")
print("kin for periphery:",kin[1249], kin[1649], kin[1330], kin[1529], "core:", kin[1022])
print("kout for periphery:",kout[1249], kout[1649], kout[1330], kout[1529], "core:", kout[1022])

print("\n5-motif admitting cross-ratio")
print("[644, 846, 655, 407, 791]")
print("kin for periphery:",kin[644], kin[846], kin[655], kin[791], kin[407], "core:", kin[1262])
print("kout for periphery:",kout[644], kout[846], kout[655], kout[791], kout[407], "core:", kout[1262], "\n\n\n")



S = similarity_matrix_cross_ratio(B)
nb_single_sources = count_single_sources(B)
nb_source_pairs = count_source_pairs(B)
nb_conserved_cross_ratios, sizes_crossratio_motifs, motifs_nodelabels, max_indegree_crossratio_motifs = \
    count_conserved_cross_ratio(B, S)

print(nb_single_sources, "\n\n", nb_source_pairs, "\n\n", nb_conserved_cross_ratios, motifs_nodelabels,
      "\n\n", max_indegree_crossratio_motifs)

# B = (B + B.T)/2

if plot_weight_matrix:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.matshow(B, aspect="auto")
    """ To look at a motif admitting a cross-ratio """
    # clique = [257, 966, 842, 1165, 782, 1806, 2131, 1560, 1498, 350, 862, 869, 1383, 2280, 487, 235, 750, 47, 1520, 1521, 1456, 2094, 311, 1784, 375, 1855]
    # clique = [644, 846, 655, 407, 791]
    # ax.matshow(B[np.ix_(clique, clique)], aspect="auto")
    """ To look at a motif admitting a two-vertex monomial """
    # print(np.where(B[:, 130:132])[0])
    # ax.matshow(B[:, 130:132], aspect="auto")
    plt.show()

""" Compute graph properties """
nb_vertices = B.shape[0]
isdirected = not is_symmetric(B)
nb_edges = count_edges_from_binary_matrix(B, isdirected=isdirected)


if draw_graph:
    g = gt.Graph(directed=isdirected)
    g.add_vertex(nb_vertices)
    if isdirected:
        I, J = np.where(B != 0)
        g.add_edge_list(zip(I, J))
    else:
        print("Hello")
        # undirected: add each edge once (upper triangle)
        iu, ju = np.triu(B, k=1).nonzero()
        g.add_edge_list(zip(iu, ju))
    print("Graph is constructed")
    nested_state = gt.minimize_nested_blockmodel_dl(g)
    print("nested state computed")
    # pos = gt.sfdp_layout(g)

    # base size and color
    vsize = g.new_vertex_property("double")
    vsize.a = 1.0

    highlight_nodes = [1851, 2054,       644, 846, 655, 407, 791,          1249, 1649, 1330, 1529]

    for idx in highlight_nodes:
        v = g.vertex(idx)
        vsize[v] = 5.0

    gt.draw_hierarchy(nested_state,
                      layout="sfdp",
                      beta=0.8,
                      # subsample_edges=5000,
                      vertex_size=vsize,
                      # hierarchy-vertex props
                      hvertex_size=0,  # no hierarchy nodes
                      hvertex_text=None,  # no labels on hierarchy
                      # hierarchy-edge props
                      hedge_pen_width=0,  # zero-width hierarchy edges
                      hedge_color=[0, 0, 0, 0],  # fully transparent RGBA
                      output="kooku1_fig4_data/connectome_pdumerilii_neuronal_network.pdf")
