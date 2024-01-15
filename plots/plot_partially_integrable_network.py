# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
from graphs.generate_integrability_partitioned_weight_matrix import integrability_partitioned_block_weight_matrix
import matplotlib.colors as mcolors
import time
import os

save_weight_matrix = True
plot_weight_matrix = False
plot_weight_distribution = False
plot_out_degree_distribution = False
draw_function = "graph_draw"

""" Generate the weight matrix """
pq = [[0.6, 0.5, 0.5, 0, 0.5],
      [0.1, 0.5, 0.3, 0, 0.1],
      [0.2, 0.01, 0.7, 0.1, 0.3],
      [0.1, 0, 0.2, 0.6, 0.3],
      [0.4, 0.5, 0, 0, 0.6]]
sizes = [38, 4, 58, 150, 250]
max_nbs_zeros_nonintegrable = sizes[0] * np.array(sizes)
max_nbs_zeros_integrable = np.tile(np.array(sizes), (len(sizes) - 1, 1))
max_nbs_zeros = np.concatenate([np.array([max_nbs_zeros_nonintegrable]), max_nbs_zeros_integrable])
proportions_of_zeros = np.array([[0.99, 0.95, 0.95, 1, 0.99],
                                 [0.99, 0.5, 0.9, 0.95, 0.95],
                                 [0.9, 0.8, 0.3, 0.8, 0.8],
                                 [0.95, 0.9, 0.7, 0.4, 0.95],
                                 [0.9, 0.95, 0.9, 0.9, 0.6]])
nbs_zeros = np.around(proportions_of_zeros * max_nbs_zeros)
nbs_zeros = nbs_zeros.astype(int)
nbs_zeros = nbs_zeros.tolist()
print(nbs_zeros)
print(max_nbs_zeros)

means = [[0, 0, 0, 0, 0],
         [0.1, 0.9, 0.5, 0.3, 0.2],
         [-0.1, -0.2, -1, -0.5, -0.3],
         [0.1, 0.1, 0.1, 0.4, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.5]]
stds = [[1, 1, 1, 1, 1],
        [0.2, 0.5, 0.5, 0.3, 0.04],
        [0.5, 0.5, 0.5, 0.5, 0.05],
        [0.5, 0.5, 0.5, 0.2, 0.03],
        [0.4, 0.5, 0.5, 0.1, 0.6]]


W = integrability_partitioned_block_weight_matrix(pq, sizes, nbs_zeros, means, stds, self_loops=True)

if save_weight_matrix:
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    directory = "/home/vincent/git_repositories/koopman-kuramoto/plots/integrability_partitioned_graph"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'{timestr}_integrability_partitioned_block_weight_matrix.npy')
    np.save(file_path, W)

if plot_weight_matrix:
    plt.matshow(W, aspect="auto")
    plt.show()

if plot_weight_distribution:
    plt.hist(W.flatten())
    plt.show()


""" Get graph tool weighted graph and its properties """


def to_graphtool_graph(weight_matrix):
    """
    Naive and slow, but explicit way, to build a graphtool weighted graph.

    IMPORTANT: graph tool adopt the convention (source, target), this is why we need to write  g.add_edge(j, i) rather
    than g.add_edge(i, j), because of our convention for the weight matrix : W_ij is the weight of the edge from j to i

    Adapted version for directed graphs of the reference below:
    carlonicolini.github.io/sections/science/2018/09/12/weighted-graph-from-adjacency-matrix-in-graph-tool.html

    Tiago Peixoto suggests:
    import graph_tool as gt
    import numpy as np
    g = gt.Graph(directed=False)
    adj = np.random.randint(0, 2, (100, 100))
    g.add_edge_list(np.transpose(adj.nonzero()))
    See Stackoverflow: create-a-weighted-graph-from-an-adjacency-matrix-in-graph-tool-python-interface
    """
    g = gt.Graph(directed=True)
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights
    num_vertices = len(weight_matrix)
    for i in range(0, num_vertices):
        for j in range(0, num_vertices):
            if weight_matrix[i, j] != 0:
                e = g.add_edge(j, i)  # See documentation
                edge_weights[e] = weight_matrix[i, j]
    return g, edge_weights


G, Edge_weights = to_graphtool_graph(W)
Deg = G.degree_property_map("in")
Deg_out = G.degree_property_map("out")
if plot_out_degree_distribution:
    plt.hist(list(Deg_out))
    plt.show()
print("deg_in", list(Deg))
print("deg_out", list(Deg_out))


""" Graph for visualization: remove selfloops, transform weights and out degrees """
W_viz = W
np.fill_diagonal(W_viz, 0)

g, weights = to_graphtool_graph(W_viz)

weights.a = np.abs(weights.a) + 0.1

deg_out = g.degree_property_map("out")
deg_out.a = 0.9 * np.sqrt(deg_out.a) + 6
print("deg_out_transformed", list(deg_out))


""" Vertices colors """
#       blue 0     orange 1    green 2     red 3     purple 4
deep = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
#   grey orange 5   pink 6  light grey 7 grey yellow 8   light_blue 9
hex_colors_short = ["#C44E52", "#8172B3", "#DD8452",  "#55A868", "#4C72B0"]
hex_colors = []
for i, size in enumerate(sizes):
    hex_colors += size*[hex_colors_short[i]]
if len(hex_colors) != g.num_vertices():
    raise ValueError("The number of colors does not match the number of vertices.")

# Convert HEX colors to RGBA
rgba_colors = [mcolors.to_rgba(c) for c in hex_colors]

# Create a property map for vertex colors
vcolor = g.new_vertex_property('vector<double>')
for v, color in zip(g.vertices(), rgba_colors):
    vcolor[v] = color


""" Edge colors """
# Normalize the weights
max_weight = max(weights.a)
min_weight = min(weights.a)
normalized_weight = g.new_edge_property('double')
for e in g.edges():
    normalized_weight[e] = (weights[e] - min_weight) / (max_weight - min_weight)

# Create a colormap from light grey to medium grey
cmap = mcolors.LinearSegmentedColormap.from_list('grey_scale', ['#e1e1e1', '#888888'])

# Map the normalized weights to colors using the colormap
edge_color = g.new_edge_property('vector<double>')
for e in g.edges():
    edge_color[e] = cmap(normalized_weight[e])


""" Curve edges (not very effective here because of random control points) """
# Create a property map for control points t
control = g.new_edge_property('vector<double>')

# Define control points for each edge
# Here, for simplicity, we use random control points
for e in g.edges():
    # For each edge, define one or two control points
    # These points should be in the format [x1, y1, x2, y2, ...]
    control[e] = np.random.random(size=6).tolist()  # Replace with your control points


""" Draw the network """
if draw_function == "graph_draw":
    gt.graph_draw(g, pos=gt.sfdp_layout(g), vertex_size=deg_out, vertex_color=vcolor,
                  vertex_fill_color=vcolor, edge_control_points=control,
                  edge_color=edge_color, edge_pen_width=weights)

else:
    state = gt.minimize_nested_blockmodel_dl(g)
    gt.draw_hierarchy(state, beta=0.95, hvertex_size=0, hedge_pen_width=0, layout="sfdp",  # layout="radial",
                      vertex_size=deg_out, vertex_color=vcolor, subsample_edges=1000,
                      vertex_fill_color=vcolor, edge_control_points=control,
                      edge_color=edge_color, edge_pen_width=weights)
    # subsample_edges=1000,
    # output=f"{timestr}_{graph_str}.svg")
