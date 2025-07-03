# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_connectomes import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import umap

# --- Load weight matrix ---
networkName = "celegans_signed"
W = get_connectome_weight_matrix(networkName)
W = W/np.linalg.norm(W, ord=2)

# --- Hierarchical clustering of rows (preserving magnitude) ---
D = pdist(W, metric='euclidean')

plt.scatter(np.arange(0, len(D)), D)
plt.show()
Z = linkage(D, method='average')  # or 'complete'

# Choose clustering threshold
threshold = 0.05
labels = fcluster(Z, t=threshold, criterion='distance')

# --- Sort rows by cluster, then by norm within cluster ---
sorted_indices = []
cluster_boundaries = []  # for visual lines
for label in np.unique(labels):
    cluster_indices = np.where(labels == label)[0]
    ordered = cluster_indices[np.argsort(np.linalg.norm(W[cluster_indices], axis=1))]
    if sorted_indices:
        cluster_boundaries.append(len(sorted_indices))  # mark boundary before appending
    sorted_indices.extend(ordered)


W_sorted = W[sorted_indices]



# # --- Apply UMAP directly to rows of W ---
# reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean', random_state=0)
# embedding = reducer.fit_transform(W)  # shape (num_rows, 2)
#
# # --- Plot 2D UMAP embedding ---
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=40, edgecolor='k')
# plt.title(f"UMAP projection of rows — {networkName}")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.grid(True)
# plt.show()



# --- Plot side-by-side matrices ---
vmax = np.max(np.abs(W))
vmin = -vmax

fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

# Left: original matrix
im0 = axes[0].matshow(W, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
axes[0].set_title("Original matrix")
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")

# Right: row-clustered matrix with lines
im1 = axes[1].matshow(W_sorted, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
axes[1].set_title("Row-clustered matrix")
axes[1].set_xlabel("Column")
axes[1].set_ylabel("Sorted rows")

# Draw lines between clusters
for b in cluster_boundaries:
    axes[1].axhline(b - 0.5, color='black', linewidth=0.8)

# Colorbar and title
fig.colorbar(im1, ax=axes, shrink=0.85, location='right', pad=0.02)
fig.suptitle(f"Weight matrix with row clustering: {networkName}", fontsize=16, y=1.04)

plt.show()
