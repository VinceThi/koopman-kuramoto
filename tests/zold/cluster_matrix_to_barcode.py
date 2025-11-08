# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from graphs.get_real_connectomes import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
# from graphs.backbone import get_backbone

def jaccard(A):
    A = A.astype(float)
    N = A.shape[0]
    D = np.diag(np.sum(A, axis=1))
    J = np.ones((N, N), dtype=float)
    return (A@A.T)*((D@J+J@D-A@A.T)**(-1))

# --- Load weight matrix ---
networkName = "pdumerilii_desmosomal"
# "cintestinalis" "celegans_signed"  "celegans" "pdumerilii_neuronal" "pdumerilii_desmosomal"
# "zebrafish meso" "mouse_meso"  "mouse_voxel"
W = get_connectome_weight_matrix(networkName)
W = W > 1
plt.scatter(W.flatten(), W.flatten(), s=4)
plt.show()
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.matshow(W, aspect="auto")
# plt.show()


S = jaccard(W > 1)
nb_sources = np.count_nonzero(np.diag(S) == 0)
print(nb_sources)

print("number of vertices involved in cross-ratios (number of identical rows) = ", np.sum(np.sum((S > 0.99999), axis=1) >= 4))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.matshow(S, aspect="auto")
plt.show()


# print(W[:100, :100])

W = W/np.linalg.norm(W, ord=2)   # Rescale matrix
W = W - np.diag(np.diag(W))  # Remove 0 elements
# TODO Use diagW later to define the natural frequencies ? omega_j - Wjj sin(alpha_jj) ? or simply set alphajj=0
zero_rows = np.all(W == 0, axis=1)
nb_zero_rows = np.sum(zero_rows)
N = len(W[:, 0])
kin = np.sum(W, axis=1)
kout = np.sum(W, axis=0)
nb_sources = np.count_nonzero(kin == 0)
nb_sinks = np.count_nonzero(kout == 0)
print(np.where(kin == 0), np.where(kout == 0))
print(f"N = {N}",
      f"\nrank(W) = {np.linalg.matrix_rank(W)}",
      f"\nnb zero rows = {nb_zero_rows}",
      f"\nnb source vertices = {nb_sources}",
      f"\nnb sink vertices = {nb_sinks}",
      f"\nnb of distances = N choose 2 = {N*(N-1)/2}")


# Is the graph connected ?
A_sym = np.abs(W) + np.abs(W.T)
degrees = np.sum(A_sym, axis=1)
L = np.diag(degrees) - A_sym
eigenvalues = np.linalg.eigvalsh(L)  # Sorted for symmetric L
num_components = np.sum(np.isclose(eigenvalues, 0, atol=1e-8))
is_connected = (num_components == 1)
print("Connected" if is_connected else f"Disconnected ({num_components} components)")

# --- Hierarchical clustering of rows (preserving magnitude) ---
D = pdist(W, metric='minkowski', p=1)

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


# --- Plot side-by-side matrices ---
vmax = np.max(np.abs(W))
vmin = -vmax

fig, axes = plt.subplots(1, 4, figsize=(20, 8), constrained_layout=True)

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
fig.colorbar(im1, ax=axes.flatten().tolist(), shrink=0.85, location='right', pad=0.02)
fig.suptitle(f"Weight matrix with row clustering: {networkName}", fontsize=16, y=1.04)

# Left: original matrix
im2 = axes[2].matshow(W != 0, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
axes[2].set_title("Original matrix (binary)")
axes[2].set_xlabel("Column")
axes[2].set_ylabel("Row")

# Right: row-clustered matrix with lines
im3 = axes[3].matshow(W_sorted != 0, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
axes[3].set_title("Row-clustered matrix (binary)")
axes[3].set_xlabel("Column")
axes[3].set_ylabel("Sorted rows")

# Draw lines between clusters
for b in cluster_boundaries:
    axes[3].axhline(b - 0.5, color='black', linewidth=0.8)

# Colorbar and title
fig.colorbar(im3, ax=axes.flatten().tolist(), shrink=0.85, location='right', pad=0.02)
fig.suptitle(f"Weight matrix with row clustering: {networkName}", fontsize=16, y=1.04)

plt.show()
