from constant_motifs.get_invariants import *
import time


# load graphml network
path = "/home/benja/Reseaux/GraphML/hermaphrodite_chemical_synapse_filtered.graphml"
network = load_graph(path, fmt='graphml')

# create all motifs associated with a constant of motion (except empty graph)
motif_3 = Graph([(0, 1), (0, 2), (0, 3)])
motif_6 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3)])
motif_9 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])
motif_complete = complete_graph(4, directed=True)
motifs_constants = [motif_3, motif_6, motif_9, motif_complete]

# create 5-star motifs
motif1_5star = Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
motif2_5star = motif1_5star.copy()
motif2_5star.add_edge(1, 0)
motif3_5star = motif2_5star.copy()
motif3_5star.add_edge(2, 0)
motif4_5star = motif3_5star.copy()
motif4_5star.add_edge(3, 0)
motif5_5star = motif4_5star.copy()
motif5_5star.add_edge(4, 0)

motifs_5star = [motif1_5star, motif2_5star, motif3_5star, motif4_5star, motif5_5star]


time1 = time.time()

# delete all self loops
network = delete_self_loops(network)

# detect motifs and extract the associated invariants
motifs_found, n, maps = motifs(network, 4, motif_list=motifs_constants, return_maps=True)
print(n)
n, invariants = extract_invariants(network, n, maps, 4)

time2 = time.time()

print(n)
print(time2 - time1)

time1 = time.time()

# detect empty motifs and extract the associated invariants
_, n_stars, maps_stars = motifs(network, 5, motif_list=motifs_5star, return_maps=True)
print(n_stars)
n_emptymotifs, invariants_emptymotifs = extract_invariants_emptymotif(network, n_stars, maps_stars, 5)

time2 = time.time()

print(n_emptymotifs)
print(time2 - time1)
