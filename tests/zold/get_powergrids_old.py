# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


""" From 
 https://arxiv.org/pdf/2402.02827
 https://figshare.com/articles/dataset/PowerGraph/22820534?file=50081700,
 https://github.com/PowerGraph-Datasets/PowerGraph-Graph/blob/main/README.md
"""
save_powergrid_adj_mat = True

# --- loader that works with MATLAB v7 and v7.3 .mat files ---
def loadmat_any(path: Path):
    try:
        from scipy.io import loadmat
        M = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in M.items() if not k.startswith("__")}
    except Exception:
        import mat73  # pip install mat73
        return mat73.loadmat(str(path))

def adjacency_dense_from_edge_index(mat_path: Path) -> np.ndarray:
    """Build dense 0/1 undirected adjacency from edge_index.mat."""
    M = loadmat_any(mat_path)
    # pick the first 2D array that looks like an edge list
    EI = None
    for v in M.values():
        arr = np.asarray(v)
        if arr.ndim == 2 and (arr.shape[1] == 2 or arr.shape[0] == 2):
            EI = arr
            break
    if EI is None:
        raise ValueError(f"No (E,2) edge list found in {mat_path}")

    EI = np.asarray(EI, dtype=int)
    if EI.shape[0] == 2 and EI.shape[1] != 2:
        EI = EI.T  # (2,E) -> (E,2)

    # MATLAB 1-based -> 0-based if needed
    if EI.min() == 1:
        EI = EI - 1

    n = int(EI.max()) + 1
    A = np.zeros((n, n), dtype=np.uint8)
    i, j = EI[:, 0], EI[:, 1]
    A[i, j] = 1
    np.fill_diagonal(A, 0)
    return A

def main():
    ROOT = Path("/graphs/powergrids/dataset_pf_opf")
    DATASETS = ["ieee24", "ieee39", "ieee118", "texas", "uk"]

    for name in DATASETS:
        raw = ROOT / name / name / "raw" / "edge_index.mat"
        if not raw.exists():
            print(f"[skip] {name}: {raw} not found")
            continue
        A = adjacency_dense_from_edge_index(raw)
        plt.matshow(A)
        plt.show()

        out_dir = raw.parent.parent / "processed_adj"   # e.g., ieee24/ieee24/processed_adj
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "adjacency.npy"

        np.save(out_path, A)
        m = int(A.sum() // 2)   # undirected edges
        print(f"[{name}] saved {out_path}  |  n={A.shape[0]}, m={m}")

if __name__ == "__main__":
    if save_powergrid_adj_mat:
        main()