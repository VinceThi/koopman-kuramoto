import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
from graphs.generate_integrability_partitioned_weight_matrix import random_weight_matrix
import matplotlib.colors as mcolors
from plots.config_rcparams import *


np.random.seed(90)

save_weight_matrix = True

# Set global plot parameters
plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False,        # Allow proper minus signs
    "font.size": 50,                   # Set default font size for plots
})

""" Generate the weight matrix """
sizes_monomial = [2, 16, 31]
sizes_crossratio = [4, 67, 50]
size_nonintegrable = [80]
sizes = np.concatenate([sizes_monomial, sizes_crossratio, size_nonintegrable])
q = len(sizes)  # Number of parts
N = sum(sizes)
print(N)
random_exponents = np.random.normal(1, 0.5, (sum(sizes_monomial), len(sizes_monomial))) + 4    # exponent magnitude
random_signs = np.random.choice([-1, 1], size=(sum(sizes_monomial), len(sizes_monomial)))    # exponent sign
random_exponents = random_exponents * random_signs

m = len(sizes_monomial)
c = len(sizes_crossratio)

probabilities_monomial = np.array([1, 0.8, 0.8])
probabilities_crossratio = np.array([[0.05, 0, 0.05, 0.05, 0.05, 0.0, 0.05],
                                     [0, 0.0, 0.02, 0.05, 0.1, 0., 0.0],
                                     [1, 0, 0.1, 0., 0.05, 0.0, 0.05],
                                     ])
probabilities_nonintegrable = [0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001]
probabilities_dict = {"monomial": probabilities_monomial, "crossratio": probabilities_crossratio,
                      "nonintegrable": probabilities_nonintegrable}
weights_monomial = np.random.normal(2, 4, (sum(sizes_monomial), sum(sizes_monomial)))
weights_crossratio = np.concatenate([np.random.normal(3, 1, (len(sizes_crossratio)-1, N)), np.random.normal(-3, 1, (1, N))])
weights_nonintegrable = np.random.normal(1, 2, (size_nonintegrable[0], N))
weights_dict = {"monomial": weights_monomial, "crossratio": weights_crossratio,
                "nonintegrable": weights_nonintegrable}

W, C = random_weight_matrix(sizes_monomial, sizes_crossratio, size_nonintegrable, random_exponents,
                            probabilities=probabilities_dict, weights=weights_dict)

if save_weight_matrix:
    np.save("random_graph_weight_matrix.npy", W)

"""" Convert to graphtool graph """
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

""" Graph for visualization: remove selfloops, transform weights and out degrees """
W_viz = W
np.fill_diagonal(W_viz, 0)

g, weights = to_graphtool_graph(W_viz)

weights.a = (np.abs(weights.a) + 0.1) / 10  # Scale weights for better visualization (edge width)

deg_out = g.degree_property_map("out")
deg_out.a = 0.9 * np.sqrt(deg_out.a) + 6


""" Vertices colors """
#       blue 0     orange 1    green 2     red 3     purple 4
# deep = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        # "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
#   grey orange 5   pink 6  light grey 7 grey yellow 8   light_blue 9
# hex_colors_short = ["#ffb38e","#ffcf9d","#ffb26f","#de8f5f","#dd8452","#87d0fa","#3c7a89","#2e4756","#16262e","#87d0fa","#3c7a89","#2e4756","#aaaaaa"]
hex_colors_short = ["#de8f5f","#ffb26f","#fcdab6","#3e6278","#4b9cb0","#87d0fa","#ffffff"]
                    # ["#C44E52", "#8172B3", "#DD8452",  "#55A868", "#4C72B0", "#FFFFFF"]
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
gt.graph_draw(g, pos=gt.sfdp_layout(g), vertex_size=14, vertex_color="#404040",
              vertex_fill_color=vcolor, edge_control_points=control,
              edge_color=edge_color, edge_pen_width=weights, output="random_graph_network.pdf")


""" Plot the eigenvalues """

eigvals = np.linalg.eigvals(W)

plt.figure(figsize=(4, 4))
plt.scatter(eigvals.real, eigvals.imag, s=10, color='#1f63b0')

# equal aspect ratio → circles look like circles
plt.gca().set_aspect('equal', 'box')

# Move the left and bottom spines to x=0 and y=0
plt.gca().spines['left'].set_position('zero')
plt.gca().spines['bottom'].set_position('zero')

# set x and y tick values
plt.yticks([-10, 10])
plt.xticks([-10, 10])

plt.xlabel(r'Re$(\lambda)$')
plt.ylabel(r'Im$(\lambda)$')
# plt.tight_layout()
plt.savefig("random_graph_eigenvalues.pdf")
plt.show()


""" Plot the singular values """
singvals = np.linalg.svd(W, compute_uv=False)

plt.figure(figsize=(4, 4))
plt.scatter(range(len(singvals)), singvals, s=10, color='#1f63b0')

plt.xlabel(r'$i$')
plt.ylabel(r'Singular value $\sigma_i$')
# plt.tight_layout()
plt.savefig("random_graph_singularvalues.pdf")
plt.show()


""" Plot the weight matrix with colored bars along the axes """
# dark grey to white to red colormap
cmap_grey_white_blue = mcolors.LinearSegmentedColormap.from_list('blue_white_orange', ["#353535ff", '#FFFFFF', "#8b0b0b"][::-1])

def add_axis_bars(ax, bars, axis='both', bar_width=0.015, offset=0.02):
    """
    Add colored bars along the axes of an imshow plot.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis containing the imshow plot
    bars : list of dict
        Each dict should have: {'start': int, 'stop': int, 'color': str}
    axis : st
        'x', 'y', or 'both' - which axes to add bars to
    bar_width : float
        Width of the bar in axis coordinates (0-1)
    offset : float
        Distance from the plot edge in axis coordinates
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    for bar in bars:
        start = bar['start']
        stop = bar['stop']
        color = bar['color']
        
        if axis in ['y', 'both']:
            # Left side vertical bar
            rect = plt.Rectangle(
                (xlim[0] - offset, start),
                bar_width * (xlim[1] - xlim[0]),
                stop - start,
                facecolor=color,
                edgecolor='none',
                clip_on=False,
                transform=ax.transData
            )
            ax.add_patch(rect)
        
        if axis in ['x', 'both']:
            # Top horizontal bar
            rect = plt.Rectangle(
                (start, ylim[1] - offset - bar_width * (ylim[1] - ylim[0])),
                stop - start,
                bar_width * (ylim[1] - ylim[0]),
                facecolor=color,
                edgecolor='none',
                clip_on=False,
                transform=ax.transData
            )
            ax.add_patch(rect)

# Plot weight matrix with colored bars
fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(W, cmap=cmap_grey_white_blue, aspect='equal', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Weight')

# generate the color bars start/stop positions and colors
bars = []
for i in range(q):
    start = sum(sizes[:i])
    stop = start + sizes[i]
    color = hex_colors_short[i]
    bars.append({'start': start, 'stop': stop, 'color': color})
# bars = [
#     {'start': 0, 'stop': 30, 'color': 'orange'},
#     {'start': 30, 'stop': 60, 'color': 'coral'},
#     {'start': 60, 'stop': 100, 'color': 'black'}
# ]

# Add the bars
add_axis_bars(ax, bars, axis='both', bar_width=0.02, offset=10)
# remove ticks
ax.set_xticks([])
ax.set_yticks([])
# ensure all axes are present
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
plt.tight_layout()

plt.savefig("random_graph_weight_matrix_with_bars.svg", dpi=400, bbox_inches='tight', pad_inches=0)
plt.show()
