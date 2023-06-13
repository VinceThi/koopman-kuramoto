from graph_tool.all import *
from graph_tool.clustering import motifs
from graph_tool.generation import complete_graph
from constant_motifs.get_invariants import extract_invariants


# create the 4 different motifs (except the empty graph)
motif_3 = Graph([(0, 1), (0, 2), (0, 3)])
motif_6 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3)])
motif_9 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])
motif_complete = complete_graph(4, directed=True)
motifs_constants = [motif_3, motif_6, motif_9, motif_complete]


#================================= TESTS FOR MOTIF_3 =================================#

# simple isomorphism
def test_motif3_1():
    g = Graph([[3, 0], [3, 1], [3, 2]])
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 1)
    print(invariants)
    assert (n, invariants) == ([1], [[0, 1, 2, 3]])

# complete graph returns no motifs
def test_motif3_2():
    g = motif_complete
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 1)
    assert (n, invariants) == ([0], [])

# motif with additional in-edges (same for all 4 vertices)
def test_motif3_3():
    g = motif_3.copy()
    g.add_edge_list([(4, 0), (4, 1), (4, 2), (4, 3)])
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 1)
    print(invariants)
    assert (n, invariants) == ([1], [[0, 1, 2, 3]])

# motif with additional in-edges (not the same for all 4 vertices)
def test_motif3_4():
    g = motif_3.copy()
    g.add_edge_list([(4, 0), (4, 2)])
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 1)
    assert (n, invariants) == ([0], [])

# motif with additional in-edges (same for all) and out-edges (should not affect the outcome)
def test_motif3_5():
    g = motif_3.copy()
    g.add_edge_list([(4, 0), (4, 1), (4, 2), (4, 3), (0, 5), (3, 5)])
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 1)
    assert (n, invariants) == ([1], [[0, 1, 2, 3]])


#================================= TESTS FOR ALL MOTIFS (EXCEPT EMPTY GRAPH) =================================#

def test_allmotifs_1():
    g = motif_3.copy()
    g.add_edge_list([(1, 3), (1, 4), (1, 5),
                     (3, 1), (3, 4), (3, 5),
                     (4, 1), (4, 3), (4, 5),
                     (5, 1), (5, 3), (5, 4),
                     (0, 4), (0, 5)])
    _, n, maps = motifs(g, 4, motif_list=motifs_constants, return_maps=True)
    print(n)
    n, invariants = extract_invariants(g, n, maps, 4)
    print(n)
    assert (n, invariants) == ([0, 0, 0, 1], [[1, 3, 4, 5]])
test_allmotifs_1()
