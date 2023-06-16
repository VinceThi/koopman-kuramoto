import graph_tool.all as gt


# get the network from Netzschleuder catalogue : https://networks.skewed.de/ 
network_name = "hermaphrodite_chemical_synapse"
network_path = f"celegans_2019/{network_name}"
network = gt.collection.ns[network_path]
print('Edge properties: ', network.edge_properties)

# save the network in GraphML format
graphml_filename = f'/home/benja/Reseaux/GraphML/{network_name}.graphml'
network.save(graphml_filename, fmt="graphml")
