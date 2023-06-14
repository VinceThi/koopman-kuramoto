import graph_tool.all as gt


# get the network from Netzschleuder catalogue : https://networks.skewed.de/ 
network = gt.collection.ns["celegansneural"]
print(network.edge_properties)

# save the network in GraphML format
graphml_filename = '/home/benja/Reseaux/GraphML/celegansneural.graphml'
network.save(graphml_filename, fmt="graphml")
