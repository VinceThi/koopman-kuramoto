
import numpy as np
from plots.config_rcparams import *
from pathlib import Path
import json
import time
import sys
from scipy.stats import binned_statistic
from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog

""" This script is not complete at all """

""" The data are generated from synchronization_map_cross_ratio_part.py """

def get_param_dictionary(filepath):
    filepath = Path(filepath)
    with filepath.open("r") as f:
        params = json.load(f)
    return params

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
path = REPO_ROOT / 'simulations' / 'kooku1_fig3_data'


file_path = Path(path / "2025_08_21_11h01min42sec_test_kuramoto_parameters_dictionary.json")
dict = get_param_dictionary(file_path)
nb_init_conditions = dict["nb_init_conditions"]
nb_source_weight = dict["nb_source_weight"]

conformist_contrarian_coupling_array = np.array(dict['conformist_contrarian_coupling_array'])
mean_module_zeta_array = np.array(dict['mean_module_zeta_array'])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax = fig.add_subplot(projection='3d')
surface_points = []
for k in np.arange(0, len(conformist_contrarian_coupling_array), 1):
    conformist_contrarian_coupling = conformist_contrarian_coupling_array[:, :, k]
    mean_module_zeta = mean_module_zeta_array[:, :, k]
    print(mean_module_zeta)
    split = 0
    left_min, left_max, n_left  = -0.6,  split, 3   # small number of bins on the left
    right_min, right_max, n_right = split, 0.6, 3  # small number of bins on the right

    edges_left  = np.linspace(left_min,  left_max,  n_left + 1)
    edges_right = np.linspace(right_min, right_max, n_right + 1)[1:]
    edges = np.r_[edges_left, edges_right]  # strictly increasing edges

    x, y = conformist_contrarian_coupling.ravel(), mean_module_zeta.ravel()
    y_mean, edges, _ = binned_statistic(x, y, statistic=lambda v: np.nanmean(v) if v.size > 0 else np.nan, bins=edges)
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    # y_std = binned_statistic(x, y, statistic=lambda v: np.nanstd(v, ddof=0) if v.size > 0 else np.nan, bins=edges)[0]
    y_lo = binned_statistic(x, y, statistic=lambda v: np.nanpercentile(v, 16), bins=edges)[0]
    y_hi = binned_statistic(x, y, statistic=lambda v: np.nanpercentile(v, 84), bins=edges)[0]
    counts = binned_statistic(x, y, statistic=lambda v: np.sum(~np.isnan(v)), bins=edges)[0]
    ax.plot(x_mid, y_mean, lw=1.5)  # , color="#aac4ff")
    ax.fill_between(x_mid, y_lo, y_hi, alpha=0.2)
    plt.show()