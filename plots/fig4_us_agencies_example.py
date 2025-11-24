# -*- coding: utf-8 -*-
# @authors: Vincent Thibeault
from pathlib import Path
from dynamics.constants_of_motion import *
from graphs.get_graph_properties import is_symmetric, count_edges_from_binary_matrix
import graph_tool.all as gt

plot_weight_matrix = 0

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "graphs" / "adjacency_matrices"

results = []

B = np.array(np.load(path/f"us_agencies__california.npy"))
kin = np.sum(B, axis=1)
kout = np.sum(B, axis=0)



""" us_agencies__california constants of motion """
# highlight_nodes = [2088, 3021, 1015, 2846,     3265, 1773, 2192, 2199, 2333,      3234, 1832, 3051, 2092, 495, 278, 1853, 861]
print("\n4-motif admitting cross-ratio")
print("[2088, 3021, 1015, 2846]")
print("kin for periphery:",   kin[2088],  kin[3021],  kin[1015],  kin[2846])
print("kout for periphery:", kout[2088], kout[3021], kout[1015], kout[2846])

print("\n5-motif (max clique) admitting cross-ratio")
print("[3265, 1773, 2192, 2199, 2333]")
print("kin for periphery:",  kin[3265],  kin[1773],  kin[2192],  kin[2199], kin[2333])
print("kout for periphery:",kout[3265], kout[1773], kout[2192], kout[2199], kout[2333])

print("\n8-motif admitting cross-ratio")
print("[3234, 1832, 3051, 2092, 495, 278, 1853, 861]")
print("kin for periphery:" , kin[3234],  kin[1832],  kin[3051],  kin[2092],  kin[495],  kin[278],  kin[1853],  kin[861])
print("kout for periphery:",kout[3234], kout[1832], kout[3051], kout[2092], kout[495], kout[278], kout[1853], kout[861], "\n\n\n")

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
    # clique = [2088, 3021, 1015, 2846]
    # clique = [3265, 1773, 2192, 2199, 2333]
    clique = [3234, 1832, 3051, 2092, 495, 278, 1853, 861]
    ax.matshow(B[np.ix_(clique, clique)], aspect="auto")
    """ To look at a motif admitting a two-vertex monomial """
    # print(np.where(B[:, 130:132])[0])
    # ax.matshow(B[:, 130:132], aspect="auto")
    plt.show()

""" Compute graph properties """
nb_vertices = B.shape[0]
isdirected = not is_symmetric(B)
nb_edges = count_edges_from_binary_matrix(B, isdirected=isdirected)


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
                    # Special motifs           maximal 5-clique                   special motif 2
highlight_nodes = [2088, 3021, 1015, 2846,    3265, 1773, 2192, 2199, 2333,      3234, 1832, 3051, 2092, 495, 278, 1853, 861]

for idx in highlight_nodes:
    v = g.vertex(idx)
    vsize[v] = 5.0

gt.draw_hierarchy(nested_state,
                  layout="sfdp",
                  beta=0.7,
                  # subsample_edges=int(4*nb_vertices//5),
                  vertex_size=vsize,
                  # hierarchy-vertex props
                  hvertex_size=0,  # no hierarchy nodes
                  hvertex_text=None,  # no labels on hierarchy
                  # hierarchy-edge props
                  hedge_pen_width=0,  # zero-width hierarchy edges
                  hedge_color=[0, 0, 0, 0],  # fully transparent RGBA
                  output="kooku1_fig4_data/us_agencies_california_network.pdf")
