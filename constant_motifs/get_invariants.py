from graph_tool.all import *
from graph_tool.clustering import motifs
from graph_tool.generation import complete_graph


# extract invariants from map
def extract_invariants(graph, n, maps, number_motifs):
    invariants = []
    for j in range(len(n)):
        for i in range(n[j]):
            # Get vertices in motif
            list_vertices = list(maps[j][i].get_array())
            # Compare in-neighbors exluding other vertices present in the motif
            in_neighbors = []
            for vertex1 in list_vertices:
                in_neighbors_temp = set([neighbour for neighbour in graph.get_in_neighbors(vertex1)])
                in_neighbors.append(in_neighbors_temp)
            diff_in_neighbors = set()
            for neighborhood in in_neighbors[1:]:
                diff_in_neighbors = diff_in_neighbors | ((in_neighbors[0] | neighborhood) - (in_neighbors[0] & neighborhood))
            diff_in_neighbors = diff_in_neighbors - set(list_vertices)
            if not diff_in_neighbors:
                invariants.append(list_vertices) # motif respects the condition on in-neighbors
            else:
                n[j] -= 1 # motif does not respect the condition on in-neighbors
    n += [0] * (number_motifs - len(n)) # zero-padding for uniform formatting
    return n, invariants

# Find the vertex that is not part of the invariant in a 5-star motif
def find_emptymotif_from_5star(graph, list_vertices):
    for vertex in list_vertices[:]:
        out_neighbors = set([neighbour for neighbour in graph.get_out_neighbors(vertex)])
        other_vertices = list_vertices.copy()
        other_vertices.remove(vertex)
        if not (set(other_vertices) - out_neighbors):
            return other_vertices
    raise RuntimeError("No center vertex was found for the 5-star.")

# extract invariants associated to 5-star motifs
def extract_invariants_emptymotif(graph, n, maps, number_motifs):
    invariants = []
    for j in range(len(n)):
        for i in range(n[j]):
            # Get 4 vertices in empty motif from 5-star motifs
            list_vertices = list(maps[j][i].get_array())
            list_vertices = find_emptymotif_from_5star(graph, list_vertices)
            # Compare in-neighbors exluding other vertices present in the motif
            in_neighbors = []
            for vertex1 in list_vertices:
                in_neighbors_temp = set([neighbour for neighbour in graph.get_in_neighbors(vertex1)])
                in_neighbors.append(in_neighbors_temp)
            diff_in_neighbors = set()
            for neighborhood in in_neighbors[1:]:
                diff_in_neighbors = diff_in_neighbors | ((in_neighbors[0] | neighborhood) - (in_neighbors[0] & neighborhood))
            diff_in_neighbors = diff_in_neighbors - set(list_vertices)
            if not diff_in_neighbors:
                invariants.append(list_vertices) # motif respects the condition on in-neighbors
            else:
                n[j] -= 1 # motif does not respect the condition on in-neighbors
    n += [0] * (number_motifs - len(n)) # zero-padding for uniform formatting
    return n, invariants

# Delete all self-loops in graph
def delete_self_loops(graph):
    for vertex in graph.vertices():
        if vertex in graph.get_out_neighbors(vertex):
            graph.remove_edge(graph.edge(vertex, vertex))
    return graph
