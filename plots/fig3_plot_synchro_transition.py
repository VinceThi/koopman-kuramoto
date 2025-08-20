
import numpy as np
from plots.config_rcparams import *
from pathlib import Path
import json
import time
import sys
from scipy.stats import binned_statistic
from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog
import seaborn as sns

def get_param_dictionary(filepath):
    filepath = Path(filepath)
    with filepath.open("r") as f:
        params = json.load(f)
    return params

SCRIPT_DIR = Path(__file__).resolve().parent  # Get current script location
REPO_ROOT = SCRIPT_DIR.parent  # Go to repo root (adjust this based on how deep your script is)
path = REPO_ROOT / 'simulations' / 'kooku1_fig3_data'  # Path to the data file

""" The data are generated from synchronization_cross_ratio_part.py """
""" Order param here is a bad terminology and means the conformists-contrarians coupling """

# Path to your JSON file
file_path = Path(path / "2025_08_19_20h31min26sec_isolated_kuramoto_parameters_dictionary.json")
dict_isolated = get_param_dictionary(file_path)
nb_init_conditions_isolated = dict_isolated["nb_init_conditions"]
order_param_array_isolated = np.array(dict_isolated['order_param_array'])
mean_module_zeta_array_isolated = np.array(dict_isolated['mean_module_zeta_array'])

file_path2 = Path(path / "2025_08_19_20h32min53sec_connected_kuramoto_parameters_dictionary.json")
dict_connected = get_param_dictionary(file_path2)
nb_init_conditions_connected = dict_connected["nb_init_conditions"]
order_param_array_connected = np.array(dict_connected['order_param_array'])
mean_module_zeta_array_connected = np.array(dict_connected['mean_module_zeta_array'])

plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})

""" Scatter plots """
# kdeplot(x=order_param_array_isolated.ravel(), y=mean_module_zeta_array_isolated.ravel())
# plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for i in range(nb_init_conditions_isolated):
    ax1.scatter(order_param_array_isolated[:, i], mean_module_zeta_array_isolated[:, i], s=5, alpha=0.6)
for i in range(nb_init_conditions_connected):
    ax2.scatter(order_param_array_connected[:, i], mean_module_zeta_array_connected[:, i], s=5, alpha=0.6)
plt.show()

""" Plot transitions """
lw = 1.5

fig, ax = plt.subplots(1, 1, figsize=(3, 2))

# for i in range(nb_init_conditions):
#     ax.scatter(order_param_array[:, i], mean_module_zeta_array[:, i], s=5, alpha=0.6)


# --- choose your split and bin counts ---
split = 0
left_min, left_max, n_left  = -0.6,  split, 10   # small number of bins on the left
right_min, right_max, n_right = split, 0.6, 200  # small number of bins on the right

# --- build edges (note [1:] to avoid duplicating 'split') ---
edges_left  = np.linspace(left_min,  left_max,  n_left + 1)
edges_right = np.linspace(right_min, right_max, n_right + 1)[1:]
edges = np.r_[edges_left, edges_right]  # strictly increasing edges

x, y = order_param_array_isolated.ravel(), mean_module_zeta_array_isolated.ravel()
y_mean, edges, _ = binned_statistic(x, y, statistic=lambda v: np.nanmean(v) if v.size > 0 else np.nan, bins=edges)
x_mid = 0.5 * (edges[:-1] + edges[1:])
# y_std  = binned_statistic(x, y, statistic=lambda v: np.nanstd(v, ddof=0) if v.size > 0 else np.nan, bins=edges)[0]
y_lo = binned_statistic(x, y, statistic=lambda v: np.nanpercentile(v, 16), bins=edges)[0]
y_hi = binned_statistic(x, y, statistic=lambda v: np.nanpercentile(v, 84), bins=edges)[0]
counts = binned_statistic(x, y, statistic=lambda v: np.sum(~np.isnan(v)), bins=edges)[0]
ax.plot(x_mid, y_mean, lw=lw)  # , color="#aac4ff")
ax.fill_between(x_mid, y_lo, y_hi, alpha=0.2)
# mask = counts >= 1 # Keep bins with data; set std=0 (or np.nan) where <2 points
# y_std[counts < 2] = np.nan  # or np.nan if you prefer gaps
# x_plot = x_mid[mask]
# y_mean_plot = y_mean[mask]
# y_std_plot = y_std[mask]
# ax.plot(x_plot, y_mean_plot, lw=2)  # , color="#aac4ff")
# ax.fill_between(x_plot, y_mean_plot - y_std_plot, y_mean_plot + y_std_plot, alpha=0.2)
ax.axvline(x=0, linestyle='--', color=deep[0])


# --- choose your split and bin counts ---
split = 0.03
left_min, left_max, n_left  = -0.6,  split, 10   # small number of bins on the left
right_min, right_max, n_right = split, 0.6, 200  # small number of bins on the right

# --- build edges (note [1:] to avoid duplicating 'split') ---
edges_left  = np.linspace(left_min,  left_max,  n_left + 1)
edges_right = np.linspace(right_min, right_max, n_right + 1)[1:]
edges = np.r_[edges_left, edges_right]  # strictly increasing edges

xc, yc = order_param_array_connected.ravel(), mean_module_zeta_array_connected.ravel()
y_meanc, edgesc, _ = binned_statistic(xc, yc, statistic=lambda v: np.nanmean(v) if v.size > 0 else np.nan, bins=edges)
x_midc = 0.5 * (edgesc[:-1] + edgesc[1:])
# y_stdc  = binned_statistic(x, y, statistic=lambda v: np.nanstd(v, ddof=0) if v.size > 0 else np.nan, bins=edgesc)[0]
y_loc = binned_statistic(xc, yc, statistic=lambda v: np.nanpercentile(v, 16), bins=edges)[0]
y_hic = binned_statistic(xc, yc, statistic=lambda v: np.nanpercentile(v, 84), bins=edges)[0]
countsc = binned_statistic(xc, yc, statistic=lambda v: np.sum(~np.isnan(v)), bins=edgesc)[0]
# maskc = countsc >= 1 # Keep bins with data; set std=0 (or np.nan) where <2 points
# y_stdc[countsc < 2] = np.nan  # or np.nan if you prefer gaps
# x_plotc = x_midc[maskc]
# y_mean_plotc = y_meanc[maskc]
# y_std_plotc = y_stdc[maskc]
# ax.plot(x_plotc, y_mean_plotc, lw=2) # , color="#fde0d0")
# ax.fill_between(x_plotc, y_mean_plotc - y_std_plotc, y_mean_plotc + y_std_plotc, alpha=0.2)
ax.plot(x_midc, y_meanc, lw=lw) # , color="#fde0d0")
ax.fill_between(x_midc, y_loc, y_hic, alpha=0.2)
ax.axvline(x=0.03, linestyle='--', color=deep[1])

ax.set_xlabel("Conformist-contrarian coupling") # $\sum_{k\in\mathcal{W}_7}W_{jk}cos(alpha_{jk})$
ax.set_ylabel("$\\langle |Z e^{-i\phi}| \\rangle_t$")
ax.set_xlim([-0.3, 0.3])
ax.set_ylim([-0.01, 1.05])

ax.scatter(-0.10931656393749649, 0.6091467474051946, color=deep[0],  zorder=10)
ax.scatter(-0.09186558132713556, 0.6495411102614075, color=deep[1],  zorder=10)
ax.scatter(-0.0444467892550075, 0.8379058278424107 , color=deep[0], marker="*",  zorder=10)
ax.scatter(-0.030035852757620923, 0.999985913021158, color=deep[1], marker="*",  zorder=10)

ax.tick_params(colors="#666666")          # tick marks and numbers
for spine in ax.spines.values():
    spine.set_color("#666666")

ax.xaxis.label.set_color("#666666")
ax.yaxis.label.set_color("#666666")
ax.title.set_color("#666666")

plt.show()

app = QApplication(sys.argv)
reply = QMessageBox.question(None, "Python", "Would you like to save the parameters, the data, and the plot?",
                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
if reply == QMessageBox.StandardButton.Yes:
    filename, ok = QInputDialog.getText(None, "File:", "Enter your file name")
    if ok:
        print("File name:", filename)
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    fig.savefig(path / f'{timestr}_{filename}_synchro_transition_crossratio_part_kuramoto.pdf')
    fig.savefig(path / f'{timestr}_{filename}_synchro_transition_crossratio_part_kuramoto.png')