from graph_tool.all import *
from graph_tool.clustering import motifs
from graph_tool.generation import complete_graph


# extract invariants from maps (WILL NEED TO BE REDONE BETTER)
def extract_invariants(graph, n, maps):
    invariants = []
    for j in range(len(n)):
        for i in range(n[j]):
            list_vertices = list(maps[j][i].get_array())
            in_neighbors = []
            for vertex1 in list_vertices:
                in_neighbors_temp = set([neighbour for neighbour in graph.get_in_neighbors(vertex1)])
                in_neighbors.append(in_neighbors_temp)
            diff_in_neighbors = set()
            for neighborhood in in_neighbors[1:]:
                diff_in_neighbors = diff_in_neighbors | ((in_neighbors[0] | neighborhood) - (in_neighbors[0] & neighborhood))
            diff_in_neighbors = diff_in_neighbors - set(list_vertices)
            if not diff_in_neighbors:
                invariants.append(list_vertices)
            else:
                n[j] -= 1
    n = list(filter((0).__ne__, n))
    return n, invariants
