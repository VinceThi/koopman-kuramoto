import numpy as np
import graph_tool.all as gt

def adjacency_matrix_to_graphtool_graph(adjacency: np.ndarray) -> gt.Graph:
    """
    Build an undirected graph-tool Graph from a symmetric boolean/0-1 adjacency matrix.
    Assumes no self-loops on the diagonal. Uses only the upper triangle for speed.
    """
    n = adjacency.shape[0]

    # (Optional) sanity checks
    # assert mask.shape[0] == mask.shape[1]
    # assert np.all(mask == mask.T)
    # assert not np.any(np.diag(mask))

    # Upper-triangular indices of True entries (edges)
    # This avoids scanning full n^2 and avoids duplicate edges.
    iu, ju = np.where(np.triu(adjacency, k=1))

    g = gt.Graph(directed=False)
    g.add_vertex(n)

    # Fastest add: pass a 2-col int array (no Python loops)
    if iu.size:
        edges = np.column_stack((iu, ju)).astype(np.int32, copy=False)
        g.add_edge_list(edges)

    return g