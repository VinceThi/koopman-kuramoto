# """ Load weight matrix """
# networkName = "pdumerilii_neuronal"
# # "cintestinalis" "celegans_signed"  "celegans" "pdumerilii_neuronal" "pdumerilii_desmosomal"
# # "zebrafish meso" "mouse_meso"  "mouse_voxel"
# # W = get_connectome_weight_matrix(networkName)
#
# # g = gt.collection.data["power"]
# g = gt.collection.data["celegansneural"]
# W = adjacency(g).toarray()
# W = W - np.diag(np.diag(W))  # Remove diagonal elements
# N = len(W[:, 0])
# # plt.scatter(np.arange(len(W.flatten())), W.flatten(), s=4)
# # plt.show()
#
# """ Binarize weight matrix """
# if np.all(np.isclose(W, 0, atol=1e-8) | np.isclose(W, 1, atol=1e-8)):
#     B = W
#     print("Weight matrix is already binary")
# else:
#     tolerance = 1e-8
#     B = W > tolerance