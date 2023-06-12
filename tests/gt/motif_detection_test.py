import pytest
from graph_tool.all import *
from graph_tool.clustering import motifs
from graph_tool.generation import complete_graph


# create the 4 different motifs (except the empty graph)
motif_3 = Graph([(0, 1), (0, 2), (0, 3)])
motif_6 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3)])
motif_9 = Graph([(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])
motif_complete = complete_graph(4, directed=True)
motifs_constants = [motif_3, motif_6, motif_9, motif_complete]

# extract invariants from maps (WILL NEED TO BE REDONE BETTER)
def extract_invariants(graph, n, maps):
    invariants = []
    for j in range(len(n)):
        for i in range(n[j]):
            list_vertices = list(maps[j][i].get_array())
            in_edges = []
            for vertex1 in list_vertices:
                in_neighbors_temp = [neighbour for neighbour in graph.get_in_neighbors(vertex1)]
                print(in_neighbors_temp)
                for vertex2 in list_vertices:
                    try:
                        in_neighbors_temp.remove(vertex2)
                    except ValueError:
                        pass
                in_edges.append(frozenset(in_neighbors_temp))
            print(set(in_edges))
            if len(set(in_edges)) <= 1:
                invariants.append(list_vertices)
            else:
                n[j] -= 1
            n = list(filter((0).__ne__, n))
    return n, invariants


# tests for motif_3

# simple isomorphism
def test_motif3_1():
    g = Graph([[3, 0], [3, 1], [3, 2]])
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps)
    print(invariants)
    assert (n, invariants) == ([1], [[0, 1, 2, 3]])

# complete graph returns no motifs
def test_motif3_2():
    g = motif_complete
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps)
    assert (n, invariants) == ([], [])

# motif with additional in-edges (same for all 4 vertices)
def test_motif3_3():
    g = motif_3.copy()
    g.add_edge_list([(4, 0), (4, 1), (4, 2), (4, 3)])
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps)
    print(invariants)
    assert (n, invariants) == ([1], [[0, 1, 2, 3]])

# motif with additional in-edges (not the same for all 4 vertices)
def test_motif3_4():
    g = motif_3.copy()
    g.add_edge_list([(4, 0), (4, 2)])
    _, n, maps = motifs(g, 4, motif_list=[motif_3], return_maps=True)
    n, invariants = extract_invariants(g, n, maps)
    assert (n, invariants) == ([], [])
