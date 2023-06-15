# Get the backbone of a graph from Netzschleuder catalogue : https://networks.skewed.de/ 
# 
# Source 1: https://gist.github.com/antoineallard/763a5773b9c5cd248a7faebcace200dd#file-import_graph_tool_extended_graphml-md
# Source 2: https://github.com/antoineallard/disparity-filter-weighted-graphs 


from pathlib import Path
import networkx as nx
import disparity_filter_weighted_graphs as dfil


graphml_filename = '/home/benja/Reseaux/GraphML/celegansneural.graphml'

# Gets the text from the GraphML file.
graphml_content = Path(graphml_filename).read_text()
# What substring should be replaced by what.
strmap = {
          'attr.type="vector_string"': 'attr.type="string"',
          'attr.type="vector_float"': 'attr.type="string"',
          'attr.type="short"': 'attr.type="long"'
          # more types could be added here
         }

# Does the subtitution.
for oldstr, newstr in strmap.items():
    graphml_content = graphml_content.replace(oldstr, newstr)
    
# The standardized GraphML content can then be directly loaded in NetworkX
graph_temp = nx.parse_graphml(graphml_content)
graph = nx.DiGraph(graph_temp)

# format weights dict
# edge_attributes = [edge[2] for edge in graph.edges(data=True)]
#print(edge_attributes)

# Compute the 'alpha' value for each edge.
dfil.compute_alpha(graph, weight="value")

# Find the optimal value for alpha. The dataframe used to find the optimal
#   value for alpha is saved to `finding_optimal_alpha.csv.zip`.
dfil.find_optimal_alpha(graph, save_optimal_alpha_data=False, method='elbow', weight="value")

# Plot the position of the optimal value for alpha.
#dfil.plot_optimal_alpha(graph)

# Create a filtered version of the original graph using the optimal value for alpha.
backbone = dfil.filter_graph(graph)

# Save the filtered version of the graph as graphml file
nx.write_graphml(backbone, graphml_filename.replace(".graphml", "_filtered.graphml"))
