# -*- coding: utf-8 -*-
# @authors: Vincent Thibeault
from pathlib import Path
from dynamics.constants_of_motion import *
from graphs.get_graph_properties import is_symmetric, count_edges_from_binary_matrix, has_tag
from tqdm import tqdm
import pandas as pd

plot_weight_matrix = 0

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "graphs" / "adjacency_matrices"

results = []
plot_weight_matrix = 0
plot_graph = 1
for file in tqdm(sorted(path.glob("*.npy"))):
    name = file.stem
    # print(name)

    # B = np.array(np.load(path/f"{name}.npy"))

    B = np.array(np.load(path/f"connectome_pdumerilii_neuronal.npy"))

    if plot_weight_matrix:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.matshow(B, aspect="auto")
        """ To look at a motif admitting a cross-ratio """
        # clique = [257, 966, 842, 1165, 782, 1806, 2131, 1560, 1498, 350, 862, 869, 1383, 2280, 487, 235, 750, 47, 1520, 1521, 1456, 2094, 311, 1784, 375, 1855]
        # ax.matshow(B[np.ix_(clique, clique)], aspect="auto")
        """ To look at a motif admitting a two-vertex monomial """
        # print(np.where(B[:, 130:132])[0])
        # ax.matshow(B[:, 130:132], aspect="auto")
        plt.show()

    """ Compute graph properties """
    nb_vertices = B.shape[0]
    isdirected = not is_symmetric(B)
    nb_edges = count_edges_from_binary_matrix(B, isdirected=isdirected)
    if name.startswith("connectome"):
        gtype = "connectome"
    elif name.startswith("powergrid"):
        gtype = "powergrid"
    else:
        if has_tag(name, "Social"):
            gtype = "social"
        elif has_tag(name, "Powergrid"):
            gtype = "powergrid"
        elif has_tag(name, "Connectome"):
            gtype = "connectome"
        else:
            raise ValueError("This network name is not associated to a valid tag (Connectome, Social, or Powergrid).")

    if plot_graph:
        import graph_tool.all as gt
        g = gt.Graph(directed=isdirected)
        g.add_vertex(nb_vertices)
        if isdirected:
            I, J = np.where(B != 0)
            g.add_edge_list(zip(I, J))
        else:
            # undirected: add each edge once (upper triangle)
            iu, ju = np.triu(B, k=1).nonzero()
            g.add_edge_list(zip(iu, ju))
        nested_state = gt.minimize_nested_blockmodel_dl(g)
        pos = gt.sfdp_layout(g)
        gt.draw_hierarchy(nested_state, pos=pos, layout="sfdp", empty_branches=True)

    """ Compute similarity matrix """
    S = similarity_matrix_cross_ratio(B)
    
    """ Count constants of motion"""
    nb_single_sources = count_single_sources(B)
    nb_source_pairs = count_source_pairs(B)[0]
    nb_conserved_cross_ratios, sizes_crossratio_motifs, motifs_nodelabels, max_indegree_crossratio_motifs =\
        count_conserved_cross_ratio(B, S)
    nb_cte_motion = count_constants_of_motion(nb_single_sources, nb_source_pairs, nb_conserved_cross_ratios, isdirected)

    """ Store results """
    results.append({
        "name": name,
        "type": gtype,
        "nb_vertices": nb_vertices,
        "nb_edges": nb_edges,
        "isdirected": isdirected,
        "nb_sources": nb_single_sources,
        "nb_2sources": nb_source_pairs,
        "nb_conserved_crossratio": nb_conserved_cross_ratios,
        "nb_cte_motion": nb_cte_motion,
        "sizes_crossratio_motifs": sizes_crossratio_motifs,
        "max_indegree_crossratio_motif": max_indegree_crossratio_motifs
    })

""" Convert to DataFrame and save """
df = pd.DataFrame(results)
out_path = ROOT / "simulations" / "kooku1_fig4_data" / "networks_constants_motion.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)

# Compute max width of each column for alignment
col_widths = {col: max(df[col].astype(str).map(len).max(), len(col)) + 2 for col in df.columns}

# Write manually with aligned columns
with open(out_path, "w") as f:
    # header
    header = "".join(col.ljust(col_widths[col]) for col in df.columns)
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    # rows
    for _, row in df.iterrows():
        line = "".join(str(row[col]).ljust(col_widths[col]) for col in df.columns)
        f.write(line + "\n")


# files = list(path.glob("*.npy"))
# print("Number of adjacency matrices:", len(files))

# For advogato: [4017, 4020, 4021, 3512]
