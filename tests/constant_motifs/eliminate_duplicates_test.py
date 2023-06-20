from constant_motifs.get_invariants import *


# create the 5-star motifs
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


# two 5-stars with the same periphery
def test_eliminate_duplicates_1():
    graph = motif1_5star.copy()
    graph.add_edge_list([(5, 1), (5, 2), (5, 3), (5, 4)])
    _, n, maps = motifs(graph, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(graph, n, maps, 5, stars=True)
    constants = eliminate_duplicates(invariants)
    assert constants == [[1, 2, 3, 4]]

def test_eliminate_duplicates_2():
    graph = motif2_5star.copy()
    graph.add_edge_list([(5, 1), (5, 2), (5, 3), (5, 4)])
    _, n, maps = motifs(graph, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(graph, n, maps, 5, stars=True)
    constants = eliminate_duplicates(invariants)
    assert constants == [[1, 2, 3, 4]]

def test_eliminate_duplicates_3():
    graph = motif3_5star.copy()
    graph.add_edge_list([(5, 1), (5, 2), (5, 3), (5, 4), (5, 0), (0, 5)])
    _, n, maps = motifs(graph, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(graph, n, maps, 5, stars=True)
    constants = eliminate_duplicates(invariants)
    assert constants == [[1, 2, 3, 4]]
