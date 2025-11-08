#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import graph_tool.all as gt
from graph_tool.spectral import adjacency as gt_adjacency

HERE = Path(__file__).resolve().parent
IN_DIR  = HERE / "datasets" / "xml.zst"
PROP_TXT = HERE / "datasets" / "datasets_properties.txt"
OUT_DIR = HERE / "adjacency_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def base_name(p: Path) -> str:
    n = p.name
    return n[:-len(".xml.zst")] if n.endswith(".xml.zst") else p.stem

def pick_scalar_weight_property(g: gt.Graph, edge_prop_names: list[str]):
    """
    From a list of candidate edge property names, return the first that is a
    scalar numeric property map (int/float/bool). Skip vectors.
    """
    # common fallbacks if edgeProp is empty or unhelpful
    fallbacks = ["weight", "value", "rating", "count", "capacity", "time", "cost", "length"]

    def is_scalar_numeric(prop) -> bool:
        t = str(prop.value_type())  # e.g. 'double', 'int32_t', 'vector<double>'
        if "vector" in t:
            return False
        return any(x in t for x in ("double", "float", "int", "long", "bool"))

    # try provided names first
    for name in edge_prop_names:
        name = name.strip()
        if not name or name == "None":
            continue
        if name in g.edge_properties:
            prop = g.edge_properties[name]
            if is_scalar_numeric(prop):
                return prop

    # otherwise try common fallbacks if present
    for name in fallbacks:
        if name in g.edge_properties and is_scalar_numeric(g.edge_properties[name]):
            return g.edge_properties[name]

    # give up -> unweighted
    return None

def main(tol=1e-8):
    # read properties table
    header = open(PROP_TXT, "r").readline().replace("#", " ").split()
    df = pd.read_table(PROP_TXT, names=header, comment="#", delimiter=r"\s+")
    df.set_index("name", inplace=True)

    files = sorted(IN_DIR.glob("*.xml.zst"))
    if not files:
        raise ValueError(f"No .xml.zst files found in {IN_DIR}")

    for fp in files:
        name = base_name(fp)
        print(f"[load] {name}  <-  {fp}")

        # row from properties
        try:
            row = df.loc[name]
        except KeyError:
            print(f"    [warn] '{name}' not found in datasets_properties.txt; treating as unweighted")
            row = pd.Series({"(un)weighted": "unweighted", "edgeProp": ""})

        g = gt.load_graph(str(fp))

        # decide weight property
        weights = None
        weighted_flag = str(row["(un)weighted"]).strip().lower() == "weighted"
        if weighted_flag:
            # parse candidate edge property names from the table
            edge_prop_col = row.get("edgeProp", "")
            candidates = [s for s in str(edge_prop_col).split(",")] if pd.notna(edge_prop_col) else []
            weights = pick_scalar_weight_property(g, candidates)
            if weights is None:
                print("    [info] no suitable scalar edge weight found; using unweighted adjacency")
                weighted_flag = False

        # build adjacency (no symmetrization)
        W = gt_adjacency(g, weight=weights).toarray()

        # binarize (your rule)
        if np.all(np.isclose(W, 0, atol=tol) | np.isclose(W, 1, atol=tol)):
            B = np.isclose(W, 1, atol=tol)
            print("    (already binary)")
        else:
            B = (np.abs(W) > tol)
        np.fill_diagonal(B, 0)

        out_path = OUT_DIR / f"{name}.npy"
        np.save(out_path, B.astype(np.bool_))
        print(f"    [saved] {out_path}  |  N={B.shape[0]}  directed={g.is_directed()}  weighted={weighted_flag}")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import graph_tool.all as gt
#
# HERE = Path(__file__).resolve().parent
# IN_DIR  = HERE / "datasets" / "xml.zst"
# PROP_TXT = HERE / "datasets" / "datasets_properties.txt"
# OUT_DIR = HERE / "adjacency_matrices"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# def base_name(p: Path) -> str:
#     # e.g. "dom__Adcock_2015a.xml.zst" -> "dom__Adcock_2015a"
#     n = p.name
#     return n[:-len(".xml.zst")] if n.endswith(".xml.zst") else p.stem
#
# def main(tol=1e-8):
#
#     header = open(PROP_TXT, "r").readline().replace("#", " ").split()
#     df = pd.read_table(PROP_TXT, names=header, comment="#", delimiter=r"\s+")
#     df.set_index("name", inplace=True)
#
#     files = sorted(IN_DIR.glob("*.xml.zst"))
#     if not files:
#         raise ValueError(f"No .xml.zst files found in {IN_DIR}")
#
#     for fp in files:
#         name = base_name(fp)
#         print(f"[load] {name}  <-  {fp}")
#
#         g = gt.load_graph(str(fp))
#
#         # Determine if graph is weighted based on table
#         is_weighted = str(df.loc[name]['(un)weighted']).strip().lower() == "weighted"
#         weights = None
#         if is_weighted:
#             # Find the edgeProp column (may be named edgeProp or egdeProp)
#             prop_field = None
#             if "edgeProp" in df.columns:
#                 prop_field = df.loc[name]["edgeProp"]
#             elif "egdeProp" in df.columns:  # sometimes misspelled
#                 prop_field = df.loc[name]["egdeProp"]
#
#             if prop_field and str(prop_field).lower() not in ("none", "", "nan"):
#                 prop_name = str(prop_field).split(",")[0].strip()
#                 if prop_name in g.edge_properties:
#                     weights = g.edge_properties[prop_name]
#                 else:
#                     print(f"    [warning] edgeProp '{prop_name}' not found in {name}; using unweighted adjacency.")
#                     weights = None
#
#         W = gt.adjacency(g, weight=weights).toarray()
#
#         # ---- Your binarization rule (no symmetrization) ----
#         if np.all(np.isclose(W, 0, atol=tol) | np.isclose(W, 1, atol=tol)):
#             B = np.isclose(W, 1, atol=tol)
#         else:
#             B = (np.abs(W) > tol)
#
#         out_path = OUT_DIR / f"{name}.npy"
#         np.save(out_path, B.astype(np.bool_))
#         print(f"    [saved] {name}  |  N={B.shape[0]}, directed={g.is_directed()}")
#
# if __name__ == "__main__":
#     main()
