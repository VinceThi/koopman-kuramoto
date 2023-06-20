from graph_tool.all import *
from graph_tool.clustering import motifs
from graph_tool.generation import complete_graph
from constant_motifs.get_constants import extract_invariants
from constant_motifs.detect_empty_motif import detect_empty_motif_inverse


# create the 4 different motifs (except the empty graph)
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

# see personal notes for graph
def test_allmotifs_1():
    g = motif_3.copy()
    g.add_edge_list([(1, 3), (1, 4), (1, 5),
                     (3, 1), (3, 4), (3, 5),
                     (4, 1), (4, 3), (4, 5),
                     (5, 1), (5, 3), (5, 4),
                     (0, 4), (0, 5)])
    _, n, maps = motifs(g, 4, motif_list=motifs_constants, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 4)
    assert (n, invariants) == ([0, 0, 0, 1], [[1, 3, 4, 5]])

# complete graph with 4 vertices
def test_allmotifs_2():
    g = complete_graph(4, directed=True)
    _, n, maps = motifs(g, 4, motif_list=motifs_constants, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 4)
    assert (n, invariants) == ([0, 0, 0, 1], [[0, 1, 2, 3]])

# complete graph with 5 vertices
def test_allmotifs_3():
    g = complete_graph(5, directed=True)
    _, n, maps = motifs(g, 4, motif_list=motifs_constants, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 4)
    assert (n, invariants.sort()) == ([0, 0, 0, 5], [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]].sort())

# complete graph with 5 vertices with motif_3
def test_allmotifs_4():
    g = complete_graph(5, directed=True)
    g.add_edge_list([(5, 6), (5, 8), (5, 7),
                     (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
                     (0, 5), (0, 6), (0, 7), (0, 8)])
    _, n, maps = motifs(g, 4, motif_list=motifs_constants, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 4)
    assert (n, invariants.sort()) == ([1, 0, 0, 5], [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]].sort())

# same as 4 but with one edge removed so that the motif_3 is no longer valid
def test_allmotifs_5():
    g = complete_graph(5, directed=True)
    g.add_edge_list([(5, 6), (5, 8), (5, 7),
                     (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
                     (0, 5), (0, 6), (0, 8)])
    _, n, maps = motifs(g, 4, motif_list=motifs_constants, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 4)
    assert (n, invariants.sort()) == ([0, 0, 0, 5], [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]].sort())


#================================= TESTS FOR EMPTY MOTIF =================================#

# simply empty graph with 4 vertices (using complete motif)
def test_emptymotif_1():
    g = Graph(4, directed=True)
    _, n, maps = detect_empty_motif_inverse(g, motif_complete)
    n, invariants = extract_invariants(g, n, maps, 1)
    assert (n, invariants) == ([1], [[0, 1, 2, 3]])

# empty graph connected to vertex 4 (star with 5 vertices)
def test_emptymotif_2():
    g = Graph([(4, 0), (4, 1), (4, 2), (4, 3)])
    _, n, maps = motifs(g, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 5, stars=True)
    assert (n, invariants) == ([1], [[0, 1, 2, 3]])

# empty graph connected to vertex 1 (star with 5 vertices)
def test_emptymotif_2():
    g = Graph([(1, 0), (1, 2), (1, 3), (1, 4)])
    _, n, maps = motifs(g, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 5, stars=True)
    assert (n, invariants) == ([1, 0, 0, 0, 0], [[0, 2, 3, 4]])

# 5-star with one edge from the periphery to the center
def test_emptymotif_3():
    g = Graph([(1, 0), (1, 2), (1, 3), (1, 4), (3, 1)])
    _, n, maps = motifs(g, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 5, stars=True)
    assert (n, invariants) == ([0, 1, 0, 0, 0], [[0, 2, 3, 4]])

# 5-star with two edges from the periphery to the center
def test_emptymotif_4():
    g = Graph([(1, 0), (1, 2), (1, 3), (1, 4), (3, 1), (0, 1)])
    _, n, maps = motifs(g, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 5, stars=True)
    assert (n, invariants) == ([0, 0, 1, 0, 0], [[0, 2, 3, 4]])

# 5-star with three edges from the periphery to the center
def test_emptymotif_5():
    g = Graph([(1, 0), (1, 2), (1, 3), (1, 4), (3, 1), (0, 1), (2, 1)])
    _, n, maps = motifs(g, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 5, stars=True)
    assert (n, invariants) == ([0, 0, 0, 1, 0], [[0, 2, 3, 4]])

# 5-star with four edges from the periphery to the center
def test_emptymotif_6():
    g = Graph([(1, 0), (1, 2), (1, 3), (1, 4), (3, 1), (0, 1), (2, 1), (4, 1)])
    _, n, maps = motifs(g, 5, motif_list=motifs_5star, return_maps=True)
    n, invariants = extract_invariants(g, n, maps, 5, stars=True)
    assert (n, invariants) == ([0, 0, 0, 0, 1], [[0, 2, 3, 4]])
