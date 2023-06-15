import graph_tool.all as gt


# get the network from Netzschleuder catalogue : https://networks.skewed.de/ 
network_name = "fly_larva"
network_path = f"{network_name}"
network = gt.collection.ns[network_path]
print('Edge properties: ', network.edge_properties)

# save the network in GraphML format
graphml_filename = f'/home/benja/Reseaux/GraphML/{network_name}.graphml'
network.save(graphml_filename, fmt="graphml")
