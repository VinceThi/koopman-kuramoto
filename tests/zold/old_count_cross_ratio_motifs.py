import numpy as np

def count_cross_ratio_motifs(S):
    """ S: similarity matrix from the function similarity_matrix_cross_ratio"""
    binarized_S = np.isclose(S, 1.0 + 0.j, atol=1e-8, rtol=1e-10)

    print(np.all(binarized_S == binarized_S.T))

    v = np.sort(np.sum(binarized_S, axis=1))
    v = v[v >= 4]
    # print(v)
    nb_cte_motion = 0
    i = 0
    while i < len(v):
        k = v[i]
        nb_cte_motion += k - 3
        i += k
        # print(k)
    return nb_cte_motion



# 3) Connected components
    # comp, hist = gt.label_components(g)
    # # comp.a (array version of comp) contains the group label of each vertex in the component
    # # (e.g., [0, 0, 0, 0, 1, 2, 2], there is three components)
    # # hist give the size of each component
    # # (e.g., [4, 1, 2])
    #
    # # 4) Count components that are cliques (fully connected inside, no outside edges)
    # nb_cross_ratio_motifs, sizes, out = 0, [], []
    # comp_ids = np.arange(len(hist))
    # for cid in comp_ids:
    #     component_size = int(hist[cid])
    #     if component_size < 4:  # hard-coded min_size
    #         continue
    #     verts = np.where(comp.a == cid)[0]
    #     deg_sum = sum(int(g.vertex(v).out_degree()) for v in verts)
    #     Ecomp = deg_sum // 2
    #     if Ecomp == component_size*(component_size - 1) // 2:  # If it is a maximal clique
    #         nb_cross_ratio_motifs += component_size - 3
    #         sizes.append(component_size)
    #         out.append(list(verts))
    #print(nb_cross_ratio_motifs, sizes, out)