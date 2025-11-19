# -*- coding: utf-8 -*-
# @authors: Vincent Thibeault
from pathlib import Path
import numpy as np
from dynamics.constants_of_motion import *
from graphs.get_graph_properties import is_symmetric, count_edges_from_binary_matrix
import graph_tool.all as gt
from tqdm import tqdm
import pandas as pd

plot_weight_matrix = 0

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "graphs" / "adjacency_matrices"

results = []

B = np.array(np.load(path/f"connectome_pdumerilii_neuronal.npy"))

if plot_weight_matrix:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.matshow(B, aspect="auto")
    """ To look at a motif admitting a cross-ratio """
    # clique = [257, 966, 842, 1165, 782, 1806, 2131, 1560, 1498, 350, 862, 869, 1383, 2280, 487, 235, 750, 47, 1520, 1521, 1456, 2094, 311, 1784, 375, 1855]
    # ax.matshow(B[np.ix_(clique, clique)], aspect="auto")
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
    # undirected: add each edge once (upper triangle)
    iu, ju = np.triu(B, k=1).nonzero()
    g.add_edge_list(zip(iu, ju))
nested_state = gt.minimize_nested_blockmodel_dl(g)
pos = gt.sfdp_layout(g)
gt.draw_hierarchy(nested_state, pos=pos, layout="sfdp", empty_branches=True)
