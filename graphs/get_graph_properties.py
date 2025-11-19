import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pathlib import Path
import pandas as pd


def is_symmetric(B, atol=1e-10):
    """
    Check if a square matrix B is symmetric within a numerical tolerance.
    """
    if B.shape[0] != B.shape[1]:
        raise ValueError("Matrix must be square to test symmetry.")
    return np.allclose(B, B.T, atol=atol)


def count_edges_from_binary_matrix(B, isdirected, atol=1e-10):
    """
    Count the number of edges from a binary matrix:
      - above the diagonal if B is symmetric,
      - in the entire matrix if B is not symmetric.
    """
    if isdirected:
        count = np.sum(B > 0.5)
    else:
        count = np.sum(np.triu(B, k=1) > 0.5)
    return int(count)


def connectivity_from_binary_matrix(B: np.ndarray):
    """
    Returns connectivity info from a dense, binary adjacency matrix B.
    """
    A = (B != 0).astype(np.int8)

    if is_symmetric(A):
        G = csr_matrix(A)

        n_strong, labels_strong = connected_components(G, directed=True, connection='strong')
        n_weak, labels_weak   = connected_components(G, directed=True, connection='weak')

        return {
            "strongly_connected": (n_strong == 1),
            "weakly_connected":   (n_weak   == 1),
            "n_strong_components": n_strong,
            "n_weak_components":   n_weak,
            "labels_strong": labels_strong,   # component id per node
            "labels_weak": labels_weak,
        }
    else:
        # ensure undirected view
        Au = ((A + A.T) > 0).astype(np.int8)
        G = csr_matrix(Au)

        n_comp, labels = connected_components(G, directed=False, connection='weak')
        return {
            "connected": (n_comp == 1),
            "n_components": n_comp,
            "labels": labels,
        }


def largest_weakly_connected_component_sparse(B: np.ndarray):
    M = csr_matrix(B.astype(bool))
    # undirected view
    U = ((M + M.T) > 0).astype(bool)
    _, labels = connected_components(U, directed=False)
    # largest label
    labs, counts = np.unique(labels, return_counts=True)
    keep_lab = labs[np.argmax(counts)]
    idx = np.where(labels == keep_lab)[0]
    idx.sort()
    # preserve directions from original B
    Bl = (M[idx][:, idx] != 0).astype(B.dtype)
    return Bl.toarray(), idx


def has_tag(name: str, tag: str, path: str = "datasets/datasets_properties.txt") -> bool:
    """
    Return True if the network with given `name` has the specified `tag`
    in the 'tags' column of datasets_properties.txt.
    """
    path = Path(__file__).resolve().parents[1] / "graphs" / "datasets" / "datasets_properties.txt"

    # Read header and data
    header = open(path, "r").readline().replace("#", " ").split()
    df = pd.read_table(path, names=header, comment="#", delimiter=r"\s+")
    df.set_index("name", inplace=True)

    if name not in df.index:
        raise ValueError(f"Network '{name}' not found in {path}")

    tags = str(df.loc[name, "tags"])  # safer than iloc[8]
    # Normalize and split tags
    tag_list = [t.strip().lower() for t in tags.split(",")]
    return tag.lower() in tag_list

# if __name__ == "__main__":
#     # ------------------ EXAMPLES ------------------
#
#     # Example for counting edges
#     B1 = np.array([[0, 1, 1],
#                    [1, 0, 0],
#                    [1, 0, 0]])  # symmetric
#
#     B2 = np.array([[0, 1, 0],
#                    [0, 0, 1],
#                    [1, 0, 0]])  # not symmetric
#
#     print(is_symmetric(B1))  # True
#     print(count_edges_from_binary_matrix(B1))  # 2 (only above diagonal)
#
#     print(is_symmetric(B2))  # False
#     print(count_edges_from_binary_matrix(B2))  # 3 (count all ones)
#
#     # Example 1: Undirected graph
#     # Two components: {0,1,2} and {3,4}
#     B_undirected = np.array([
#         [0,1,1,0,0],
#         [1,0,1,0,0],
#         [1,1,0,0,0],
#         [0,0,0,0,1],
#         [0,0,0,1,0]
#     ])
#
#     Bl1, idx1 = largest_weakly_connected_component_sparse(B_undirected)
#     print("UNDIRECTED EXAMPLE")
#     print("Largest component nodes:", idx1)
#     print("Submatrix:\n", Bl1)
#
#     # Example 2: Directed graph
#     # 0→1→2 forms a chain, 3↔4 is a separate small SCC
#     B_directed = np.array([
#         [0,1,0,0,0],
#         [0,0,1,0,0],
#         [0,0,0,0,0],
#         [0,0,0,0,1],
#         [0,0,0,1,0]
#     ])
#
#     Bl2, idx2 = largest_weakly_connected_component_sparse(B_directed)
#     print("\nDIRECTED EXAMPLE")
#     print("Largest component nodes:", idx2)
#     print("Submatrix:\n", Bl2)