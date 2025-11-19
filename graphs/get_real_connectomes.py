# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
import warnings
from pathlib import Path
from graphs.get_graph_properties import largest_weakly_connected_component_sparse
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

def get_connectome_weight_matrix(graph_name):
    """
    Return the weight matrix for a given graph.
    graph_name (str): "mouse_meso", "zebrafish_meso", "celegans_yan_et_al_2017",
                      "celegans_signed", "drosophila_hemibrain", "ciona",
                      "platynereis_dumerilii_neuronal"
    """
    path = Path(__file__).parent / "connectomes"

    if graph_name == "celegans_yan_et_al_2017":
        # Data obtained from Mohamed Bahdine, extracted as described in the
        # supplementary material of the article : Network control principles
        # predict neuron function in the C. elegans connectome - Yan et al.
        # The data come from Wormatlas.
        A = np.array(1 * np.load(path/"C_Elegans.npy"))
        # N = 279
        # rank_celegans = 273

    elif graph_name == "celegans_signed":
        # Data: https://elegansign.linkgroup.hu/#!NT+R%20method%20prediction
        # Paper: https://doi.org/10.1371/journal.pcbi.1007974
        df = pd.read_excel(path/'celegans_weighted_directed_signed.xls', usecols="A,D,E,P")
        df = df.replace(to_replace=['+', '-', 'no pred', 'complex'],
                        value=[1, -1, 0, 0])
        df_dale = df[["Source"]]
        df_dale["Strength x Sign"] = df["Edge Weight"] * df["Sign"]
        df_dale = df_dale.groupby(['Source']).sum()
        # We complete the missing data using Dale's principle, i.e., if most
        # of the synapses of a neuron are excitatory (inhibitory), then we
        #  consider that the unknown ones are excitatory (inhibitory). When no
        # information allows to apply Dale's principle, we consider the neuron
        # as an excitator given the fact that there are more excitators than
        # inhibitors in connectomes.
        for i, neuron in enumerate(df["Source"]):
            if df["Sign"].values[i] == 0 \
                    and df_dale.loc[neuron].values[0] >= 0:
                df.loc[i, ['Sign']] = 1
            elif df["Sign"].values[i] == 0 \
                    and df_dale.loc[neuron].values[0] < 0:
                df.loc[i, ['Sign']] = -1
        df["Weight x Sign"] = df["Edge Weight"] * df["Sign"]
        # with pd.option_context('display.max_rows', None,
        #                        'display.max_columns', None):
        #     print(df)
        # print(df["Weight x Sign"].sum())
        G_celegans = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr='Weight x Sign',
                                             create_using=nx.DiGraph())
        A = nx.to_numpy_array(G_celegans, weight='Weight x Sign')
        # N = 297

    elif graph_name == "drosophila_hemibrain":
        df = pd.read_csv(
            path/'drosophila_exported-traced-adjacencies-v1.1/'
                       'traced-total-connections.csv')
        Graphtype = nx.DiGraph()
        G_drosophila = nx.from_pandas_edgelist(df, source='bodyId_pre', target='bodyId_post', edge_attr='weight',
                                               create_using=Graphtype)
        A = nx.to_numpy_array(G_drosophila, weight='weight')
        # N = 21733
        # srank = 11.5811

    elif graph_name == "cintestinalis":
        A_from_xlsx = pd.read_excel(path/'ciona_intestinalis_lavaire_elife-16962-fig16-data1-v1_modified.xlsx').values
        A_ciona_nan = np.array(A_from_xlsx[0:, 1:])
        A_ciona = np.array(A_ciona_nan, dtype=float)
        where_are_NaNs = np.isnan(A_ciona)
        A_ciona[where_are_NaNs] = 0
        A = A_ciona
        # A = (A_ciona > 0).astype(float)
        # N = 213
        # rank = 203

    elif graph_name == "mouse_meso":
        # Oh, S., Harris, J., Ng, L. et al.
        # A mesoscale connectome of the mouse brain.
        # Nature 508, 207–214 (2014) doi:10.1038/nature13186
        # To binary matrix (with "> 0")
        # A = (np.loadtxt(path_str + "ABA_weight_mouse.txt") > 0).astype(float)
        A = (np.loadtxt(path/"ABA_weight_mouse.txt")).astype(float)
        # N = 213
        # rank = 185

    elif graph_name == "mouse_voxel":
        # Coletta et al., "Network structure of the mouse brain
        #  connectome with voxel resolution"

        dictionary = scipy.io.loadmat(path/'full_connectome_no_thr.mat')
        A = dictionary['full_connectome_no_thr']

    elif graph_name == "zebrafish_meso":
        # Kunst et al.
        # "A Cellular-Resolution Atlas of the Larval Zebrafish Brain",
        # (2019) with the treatment of Antoine Légaré
        # We do not have exactly the same regions than the ones in the paper
        # where the matrix is 36 by 36. Here, we have 71 regions that are
        # mutually exclusive and collectively exhaustive (these are the terms
        # of the corresponding author of the above paper), in the sense that
        # it covers the whole volume without overlap

        df = pd.read_csv(path/'Connectivity_matrix_zebra_fish_mesoscopic.csv')
        dictio = {'X': 0}  # We put zeros temporarily on the diagonal
        df = df.replace(dictio)

        volumes = np.array(
            1 * np.load(path/"volumes_zebrafish_meso.npy"))
        relativeVolumes = volumes / sum(volumes)
        adjacency = df.to_numpy()[:, 1:-1].astype(float)
        # """ To get an undirected graph """
        # for i in range(adjacency.shape[0]):
        #     for j in range(i+1, adjacency.shape[0]):
        #         adjacency[i, j] = (adjacency[i, j] + adjacency[j, i]) /
        #  (relativeVolumes[i] + relativeVolumes[j])
        #         adjacency[j, i] = adjacency[i, j]
        """ To get a directed graph """
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[0]):
                adjacency[i, j] = adjacency[i, j] / (
                        relativeVolumes[i] + relativeVolumes[j])
        adjacency = adjacency / np.amax(adjacency)
        adjacency = np.log(adjacency + 0.00001)
        adjacency -= np.amin(adjacency)
        adjacency = adjacency / np.amax(adjacency)
        # We add a diagonal because there are interactions within modules
        A = adjacency + np.eye(len(adjacency[0]))
        # N = 71
        # rank_zebrafish_meso = 71

    elif graph_name == "pdumerilii_neuronal":
        G_platynereis = nx.read_graphml(path/"pdumerilii_neuronal.xml")
        A = nx.to_numpy_array(G_platynereis)

    elif graph_name == "pdumerilii_desmosomal":
        G_platynereis = \
            nx.read_graphml(path/"pdumerilii_desmosomal.xml",
                            force_multigraph=True)
        A = nx.to_numpy_array(G_platynereis,
                              nodelist=sorted(G_platynereis.nodes()),
                              multigraph_weight=sum)
        """
         Note: There are repeated edges in the dataset.
         The number of edges, confirmed with G. Jékely, is 5455 and not 6961.
        """
        # print(np.all(A.T == A), np.sum(np.triu((A > 0).astype(float), 1)),
        #       np.sum(np.triu(A)), np.sum(np.diag((A > 0).astype(float))))

    else:
        raise ValueError(f"This graph_str ({graph_name}) connectome is not an option. "
                         "See the documentation of "
                         "get_connectome_weight_matrix")

    return A


def main():

    DATASETS = ["celegans_yan_et_al_2017", "cintestinalis", "pdumerilii_neuronal", "pdumerilii_desmosomal",
                "zebrafish_meso", "mouse_meso", "mouse_voxel", "drosophila_hemibrain"]

    for name in DATASETS:

        W = get_connectome_weight_matrix(name)

        """ Binarize weight matrix """
        if np.all(np.isclose(W, 0, atol=1e-8) | np.isclose(W, 1, atol=1e-8)):
            B = W
            print("Weight matrix is already binary")
        else:
            tolerance = 1e-8
            B = np.abs(W) > tolerance
        np.fill_diagonal(B, 0)
        # plt.matshow(B, aspect="auto")
        # plt.show()
        B = largest_weakly_connected_component_sparse(B)[0]

        out_dir = Path(__file__).parent / "adjacency_matrices"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(name)
        out_path = out_dir / f"connectome_{name}.npy"
        np.save(out_path, B)

        print(f"[{name}] saved {out_path}  |  N={B.shape[0]}")

if __name__ == "__main__":
    main()