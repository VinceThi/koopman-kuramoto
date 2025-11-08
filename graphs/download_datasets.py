# -*- coding: utf-8 -*-
# @author: Antoine Allard <antoineallard.info> and Vincent Thibeault

import json
import urllib.request
import pandas as pd
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
import time
from tqdm import tqdm

# ----- paths (absolute, script-relative) -----
SCRIPT_DIR = Path(__file__).resolve().parent                 # .../koopman-kuramoto/graphs
DATA_ROOT  = (SCRIPT_DIR / "datasets").resolve()             # .../koopman-kuramoto/graphs/datasets
GRAPH_PROP_PATH = DATA_ROOT / "datasets_properties.txt"

# optional local import for your fancy table
sys.path.append('pandas-fancy-table-io')
from pandas_fancy_table_io import to_fancy_table

# Adds the custom method to Pandas.
pd.DataFrame.to_fancy_table = to_fancy_table

# ----- HTTP setup: browser-like UA + helper with retries -----
opener = urllib.request.build_opener()
opener.addheaders = [
    ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    ("Accept", "*/*"),
]
urllib.request.install_opener(opener)


def download_file(source_url: str, target_path: Path, tries: int = 3, backoff: float = 1.5, timeout: int = 60):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(source_url, timeout=timeout) as r, open(str(target_path), "wb") as f:
                f.write(r.read())
            return
        except (HTTPError, URLError) as e:
            last_err = e
            if k < tries - 1:
                time.sleep(backoff ** k)
            else:
                raise last_err


def analyze_graph(graphDict, graphDictTags):
    nbVertices = graphDict['num_vertices']
    nbEdges = graphDict['num_edges']

    direction = 'undirected'

    if graphDict['is_directed']:
        direction = 'directed'

    partite = 'unipartite'
    if graphDict['is_bipartite']:
        partite = 'bipartite'

    weights = 'unweighted'
    if 'Weighted' in graphDictTags:
        weights = 'weighted'

    tags = list(graphDictTags)
    for tag in ['Weighted', 'Unweighted', 'Metadata', 'Multigraph']:
        if tag in tags:
            tags.remove(tag)
    tags = ','.join(tag for tag in tags).replace(' ', '')

    vps = ','.join(ep[0] for ep in graphDict.get('vertex_properties', []))
    eps = ','.join(ep[0] for ep in graphDict.get('edge_properties', []))

    return [direction, weights, partite, nbVertices, nbEdges, tags, vps, eps]


def has_desired_tags(tag_list):
    """Return True iff tags match:
       - 'Social' AND NOT ('Animal' or 'Sport'), OR
       - 'Connectome', OR
       - 'Power grid'."""
    tags = set(tag_list)
    social_ok = ("Social" in tags) and not (("Animal" in tags) or ("Sport" in tags))
    return social_ok or ("Connectome" in tags) or ("Power grid" in tags)


def download_datasets():
    print("Saving to:", DATA_ROOT, flush=True)

    fileExtensions = ['xml.zst']

    # Skip very large/unavailable collections
    blacklist = [
        'openstreetmap', 'human_brains', 'moviegalaxies', 'route_views',
        'internet_top_pop', 'eu_procurements_alt', 'add_health', 'hiv_transmission'
    ]

    netzschleuderURL = 'https://networks.skewed.de'

    header = ['name', '(un)dir', '(un)weighted', 'uni/bi-partite', 'nbVertices', 'nbEdges',
              'density', 'averageDegree', 'tags', 'vertexProp', 'edgeProp']   # typo fixed

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # List all available graphs
    with urllib.request.urlopen(netzschleuderURL + '/api/nets', timeout=60) as f:
        availableGraphs = json.loads(f.read())

    graphPropList = []

    for graphEntry1 in tqdm(availableGraphs):
        if graphEntry1 in blacklist:
            continue

        with urllib.request.urlopen(netzschleuderURL + '/api/net/' + graphEntry1, timeout=60) as f:
            graphDict = json.loads(f.read())

        if not has_desired_tags(graphDict.get('tags', [])):
            continue

        for graphEntry2 in graphDict['nets']:

            if graphEntry1 == graphEntry2:
                networkName = graphEntry1.replace('.', '_')
                sourceRoot = f"{netzschleuderURL}/net/{graphEntry1}/files/{graphEntry1}"
                graphMetricsDict = graphDict['analyses']
            else:
                networkName = (graphEntry1 + '__' + graphEntry2).replace('.', '_')
                sourceRoot = f"{netzschleuderURL}/net/{graphEntry1}/files/{graphEntry2}"
                graphMetricsDict = graphDict['analyses'][graphEntry2]

            sourceRoot = sourceRoot.replace(' ', '%20')

            nbVertices = graphMetricsDict['num_vertices']
            nbEdges    = graphMetricsDict['num_edges']
            isDirected = graphMetricsDict['is_directed']
            isBipartite= graphMetricsDict['is_bipartite']

            # FILTERS
            conditions = [
                nbVertices > 100,
                nbVertices < 10000,
            ]
            if not all(conditions):
                continue

            graphPropList.append([networkName] + analyze_graph(graphMetricsDict, graphDict['tags']))

            safe_name = (networkName.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', ''))

            for fileExtension in fileExtensions:
                source = f"{sourceRoot}.{fileExtension}"
                target = DATA_ROOT / fileExtension / f"{safe_name}.{fileExtension}"   # ✅ graphs/datasets/...

                if not target.exists():
                    print('downloading', source, '->', target, flush=True)
                    download_file(source, target)
                # else:
                #     print('skip (exists):', target, flush=True)

    # Save the summary table
    graphPropDF = pd.DataFrame(graphPropList, columns=header)
    graphPropDF.sort_values('name', inplace=True)
    graphPropDF.reset_index(drop=True, inplace=True)
    GRAPH_PROP_PATH.parent.mkdir(parents=True, exist_ok=True)
    graphPropDF.to_fancy_table(str(GRAPH_PROP_PATH),
                               tabulateprops={'stralign': 'right', 'colalign': ('left',)})

if __name__ == "__main__":
    download_datasets()
