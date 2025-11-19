# -*- coding: utf-8 -*-
# @author: Vincent Thibeault
#!/usr/bin/env python3
from pathlib import Path
import re
import urllib.request
import time
import numpy as np
import matplotlib.pyplot as plt
from graphs.get_graph_properties import largest_weakly_connected_component_sparse


"""
The data are from MATPOWER:
R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, "MATPOWER:
  Steady-State Operations, Planning and Analysis Tools for Power Systems
  Research and Education," Power Systems, IEEE Transactions on, vol. 26,
  no. 1, pp. 12-19, Feb. 2011.
  doi: 10.1109/TPWRS.2010.2051168
  
and are available on GitHub: https://github.com/MATPOWER/matpower/tree/master/data
"""

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "adjacency_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of cases desired
CASES = [
    "case10ba.m",
    "case118.m",
    "case118zh.m",
    "case1197.m",
    "case12da.m",
    "case1354pegase.m",
    "case13659pegase.m",
    "case136ma.m",
    "case14.m",
    "case141.m",
    "case145.m",
    "case15da.m",
    "case15nbr.m",
    "case16am.m",
    "case16ci.m",
    "case17me.m",
    "case18.m",
    "case1888rte.m",
    "case18nbr.m",
    "case1951rte.m",
    "case22.m",
    "case2383wp.m",
    "case24_ieee_rts.m",
    "case2736sp.m",
    "case2737sop.m",
    "case2746wop.m",
    "case2746wp.m",
    "case2848rte.m",
    "case2868rte.m",
    "case2869pegase.m",
    "case28da.m",
    "case30.m",
    "case300.m",
    "case3012wp.m",
    "case30Q.m",
    "case30pwl.m",
    "case3120sp.m",
    "case3375wp.m",
    "case33bw.m",
    "case33mg.m",
    "case34sa.m",
    "case38si.m",
    "case39.m",
    # "case4_dist.m",
    # "case4gs.m",
    # "case5.m",
    "case51ga.m",
    "case51he.m",
    "case533mt_hi.m",
    "case533mt_lo.m",
    "case57.m",
    "case59.m",
    "case60nordic.m",
    "case6468rte.m",
    "case6470rte.m",
    "case6495rte.m",
    "case6515rte.m",
    "case69.m",
    # "case6ww.m",
    "case70da.m",
    "case74ds.m",
    "case8387pegase.m",
    "case85.m",
    "case89pegase.m",
    # "case9.m",
    "case9241pegase.m",
    "case94pi.m",
    # "case9Q.m",
    # "case9target.m",
    "case_ACTIVSg10k.m",
    "case_ACTIVSg200.m",
    "case_ACTIVSg2000.m",
    "case_ACTIVSg25k.m",
    "case_ACTIVSg500.m",
    # "case_ACTIVSg70k.m",
    "case_RTS_GMLC.m",
    # "case_SyntheticUSA.m",
    "case_ieee30.m"
]

# GitHub raw URL template for MATPOWER data
RAW_URL = "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/{fname}"

# ----- Utilities -----
def _get(url, tries=3, backoff=1.7, timeout=30) -> str:
    """Download text with a browser-like User-Agent and simple retries."""
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        ("Accept", "*/*"),
    ]
    urllib.request.install_opener(opener)
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            last = e
            if k < tries - 1:
                time.sleep(backoff ** k)
    raise last


def _extract_block(text: str, var_name: str) -> str:
    """
    Extract the MATLAB matrix literal after 'var_name = [' ... '];'
    Returns the inside text (without the wrapping brackets), or raises if not found.
    """
    # robust regex across line breaks and comments
    # matches: var_name = [  ... ];   (non-greedy inside)
    pat = re.compile(rf"{re.escape(var_name)}\s*=\s*\[\s*(.*?)\s*\];", re.S)
    m = pat.search(text)
    if not m:
        raise ValueError(f"Could not find '{var_name} = [ ... ];' block")
    return m.group(1)


def _rows_to_array(block: str) -> np.ndarray:
    """
    Convert inside of a MATLAB matrix block to a float array.
    Splits by ';' into rows, then whitespace into columns.
    Ignores empty/comment-only rows.
    """
    rows = []
    for raw in block.split(';'):
        line = raw.strip()
        if not line:
            continue
        # drop comments starting with % (MATLAB)
        line = line.split('%', 1)[0].strip()
        if not line:
            continue
        # split on whitespace and convert to float
        vals = [float(x) for x in line.split()]
        rows.append(vals)
    if not rows:
        raise ValueError("Matrix block parsed empty")
    arr = np.array(rows, dtype=float)
    return arr


def parse_branch_table(matpower_text: str) -> np.ndarray:
    """
    Parse 'mpc.branch = [ ... ];' into a float array of shape (E, >=13).
    Columns (MATPOWER):
      0 fbus, 1 tbus, 2 r, 3 x, 4 b, 5 rateA, 6 rateB, 7 rateC, 8 ratio,
      9 angle, 10 status, 11 angmin, 12 angmax
    """
    block = _extract_block(matpower_text, "mpc.branch")
    arr = _rows_to_array(block)
    if arr.shape[1] < 11:
        raise ValueError(f"branch table has {arr.shape[1]} columns (< 11 needed for status)")
    return arr


def build_adjacency_from_branch(branch: np.ndarray, directed: bool = True) -> np.ndarray:
    """
    Build binary adjacency matrix A from MATPOWER's data branch tables:
    - keep only rows with status == 1
    - buses are 1-based in files; convert to 0-based indices
    - directed=True
    Returns A as uint8 (0/1).
    """
    # Check status of the branch (column 11th = index 10)
    on = branch[:, 10] != 0
    edges = branch[on, :2].astype(int)  # fbus, tbus
    if edges.size == 0:
        raise ValueError("No in-service branches (status==1) found")

    # MATPOWER buses are labeled starting from 1
    edges -= 1
    n = int(edges.max()) + 1

    A = np.zeros((n, n), dtype=np.uint8)
    i = edges[:, 0]
    j = edges[:, 1]

    A[i, j] = 1
    if not directed:
        A[j, i] = 1

    np.fill_diagonal(A, 0)
    return largest_weakly_connected_component_sparse(A)[0]



def main():
    for fname in CASES:
        url = RAW_URL.format(fname=fname)
        print(f"[fetch] {fname}  <-  {url}")
        txt = _get(url)

        # Parse branch table and build adjacency
        branch = parse_branch_table(txt)
        A = build_adjacency_from_branch(branch)

        # plt.matshow(A)
        # plt.show()

        # Save
        stem = Path(fname).stem  # e.g. "case10ba"
        out = OUT_DIR / f"powergrid_{stem}.npy"
        np.save(out, A)
        m = int(A.sum())
        print(f"  -> saved {out}   |  n={A.shape[0]}  edges={m}")




""" Other data from: 
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


def main2():

    ROOT = Path("/graphs/powergrids/dataset_pf_opf")
    DATASETS = ["texas", "uk"]   #  "ieee24", "ieee39", "ieee118" already in MATPOWER

    for name in DATASETS:
        raw = ROOT / name / name / "raw" / "edge_index.mat"
        if not raw.exists():
            print(f"[skip] {name}: {raw} not found")
            continue
        A = adjacency_dense_from_edge_index(raw)
        plt.matshow(A)
        plt.show()

        out_dir = raw.parent.parent / "processed_adj"   # e.g., ieee24/ieee24/processed_adj # IMP: It does not go in adjacency_matrices folder directly here
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "adjacency.npy"
        if save_powergrid_adj_mat:
            np.save(out_path, A)
            m = int(A.sum() // 2)   # undirected edges
            print(f"[{name}] saved {out_path}  |  n={A.shape[0]}, m={m}")


if __name__ == "__main__":
   main()