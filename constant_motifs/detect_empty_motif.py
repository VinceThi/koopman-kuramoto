import numpy as np
import time
from constant_motifs.get_invariants import *
from graph_tool.spectral import adjacency


def detect_empty_motif(network, complete_motif):
    adj = adjacency(network).toarray().astype(np.int8)
    inverted_adj = ~adj + 2 - np.identity(adj.shape[0])
    network_temp = Graph(np.array(np.nonzero(inverted_adj)).T)
    motifs_found, n, maps = motifs(network_temp, 4, motif_list=[complete_motif], return_maps=True)
    return motifs_found, n, maps
