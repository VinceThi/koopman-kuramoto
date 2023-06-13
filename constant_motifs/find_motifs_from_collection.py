from constant_motifs.get_invariants import *
from constant_motifs.detect_empty_motif import detect_empty_motif
import time


# get desired network from Netzschleuder: https://networks.skewed.de/
network = collection.ns["celegansneural"]

# create all motifs associated with a constant of motion (except empty graph)
motif_3 = Graph([(0, 1), (0, 2), (0, 3)])
motif_6 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3)])
motif_9 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])
motif_complete = complete_graph(4, directed=True)
motifs_constants = [motif_3, motif_6, motif_9, motif_complete]

time1 = time.time()

# detect motifs and extract the associated invariants
motifs_found, n, maps = motifs(network, 4, motif_list=motifs_constants, return_maps=True)
print(n)
n, invariants = extract_invariants(network, n, maps, 4)

time2 = time.time()

print(n)
print(time2 - time1)

time1 = time.time()

# detect empty motifs and extract the associated invariants (TAKES TOO LONG)
# motifs_found, n, maps = detect_empty_motif(network, motif_complete)
# print(n)
# n, invariants = extract_invariants(network, n, maps, 4)

# time2 = time.time()

# print(n)
# print(time2 - time1)
