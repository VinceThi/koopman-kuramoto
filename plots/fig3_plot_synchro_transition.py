
import numpy as np
from plots.config_rcparams import *
from pathlib import Path
import json
import time
import sys
from scipy.stats import binned_statistic
from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog

scatter_plots = False

def get_param_dictionary(filepath):
    filepath = Path(filepath)
    with filepath.open("r") as f:
        params = json.load(f)
    return params

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
path = REPO_ROOT / 'simulations' / 'kooku1_fig3_data'

""" The data are generated from synchronization_cross_ratio_part.py """

file_path = Path(path / "2025_08_19_20h31min26sec_isolated_kuramoto_parameters_dictionary.json")
dict_isolated = get_param_dictionary(file_path)
nb_init_conditions_isolated = dict_isolated["nb_init_conditions"]
conformist_contrarian_coupling_array_isolated = np.array(dict_isolated['order_param_array'])
mean_module_zeta_array_isolated = np.array(dict_isolated['mean_module_zeta_array'])

file_path2 = Path(path / "2025_08_21_07h39min51sec_5_kuramoto_parameters_dictionary.json")
dict_connected = get_param_dictionary(file_path2)
nb_init_conditions_connected = dict_connected["nb_init_conditions"]
conformist_contrarian_coupling_array_connected = np.array(dict_connected['conformist_contrarian_coupling_array'])
mean_module_zeta_array_connected = np.array(dict_connected['mean_module_zeta_array'])


file_path3= Path(path / "2025_08_21_07h39min23sec_20_kuramoto_parameters_dictionary.json")
dict_connected2 = get_param_dictionary(file_path3)
nb_init_conditions_connected2 = dict_connected2["nb_init_conditions"]
conformist_contrarian_coupling_array_connected2 = np.array(dict_connected2['conformist_contrarian_coupling_array'])
mean_module_zeta_array_connected2 = np.array(dict_connected2['mean_module_zeta_array'])

plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})

""" Scatter plots """
if scatter_plots:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    # for i in range(nb_init_conditions_isolated):
    #     ax1.scatter(conformist_contrarian_coupling_array_isolated[:, i], mean_module_zeta_array_isolated[:, i], s=5, alpha=0.6)
    # ax1.set_title("$w_s = 0$")
    # ax1.set_xlabel("Conformist-contrarian coupling")
    # ax1.set_ylabel("Order parameter $\\langle |Z e^{-i\phi}| \\rangle_t$")
    # for i in range(nb_init_conditions_connected):
    #     ax2.scatter(conformist_contrarian_coupling_array_connected[:, i], mean_module_zeta_array_connected[:, i], s=5, alpha=0.6)
    # ax2.set_title("$w_s = 5$")
    # ax2.set_xlabel("Conformist-contrarian coupling")
    # ax2.set_ylabel("Order parameter $\\langle |Z e^{-i\phi}| \\rangle_t$")
    for i in range(nb_init_conditions_connected):
        ax3.scatter(conformist_contrarian_coupling_array_connected2[:, i], mean_module_zeta_array_connected2[:, i], s=5, alpha=0.6)
    ax3.set_title("$w_s = 20$")
    ax3.set_xlabel("Conformist-contrarian coupling")
    ax3.set_ylabel("Order parameter $\\langle |Z e^{-i\phi}| \\rangle_t$")
    plt.show()

""" Plot transitions """
lw = 1.5

fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.3))

# for i in range(nb_init_conditions):
#     ax.scatter(conformist_contrarian_coupling_array[:, i], mean_module_zeta_array[:, i], s=5, alpha=0.6)


""" source weight = 0 """
split = 0
left_min, left_max, n_left  = -0.6,  split, 20   # small number of bins on the left
right_min, right_max, n_right = split, 0.6, 200  # small number of bins on the right

edges_left  = np.linspace(left_min,  left_max,  n_left + 1)
edges_right = np.linspace(right_min, right_max, n_right + 1)[1:]
edges = np.r_[edges_left, edges_right]  # strictly increasing edges

x, y = conformist_contrarian_coupling_array_isolated.ravel(), mean_module_zeta_array_isolated.ravel()
y_mean, edges, _ = binned_statistic(x, y, statistic=lambda v: np.nanmean(v) if v.size > 0 else np.nan, bins=edges)
x_mid = 0.5 * (edges[:-1] + edges[1:])
y_lo = binned_statistic(x, y, statistic=lambda v: np.nanpercentile(v, 16), bins=edges)[0]
y_hi = binned_statistic(x, y, statistic=lambda v: np.nanpercentile(v, 84), bins=edges)[0]
counts = binned_statistic(x, y, statistic=lambda v: np.sum(~np.isnan(v)), bins=edges)[0]
ax.plot(x_mid, y_mean, lw=lw, label="$\mathcal{A}_s = 0$", color="#aac4ff")
ax.fill_between(x_mid, y_lo, y_hi, alpha=0.1, color="#aac4ff")
# ax.axvline(x=0, ymax=1, linestyle='--', color="#aac4ff", linewidth=1)


""" source weight = 5 """
# --- choose your split and bin counts ---
# split = 0
# left_min, left_max, n_left  = -0.6,  split, 20   # small number of bins on the left
# right_min, right_max, n_right = split, 0.6, 200  # small number of bins on the right
#
# # --- build edges (note [1:] to avoid duplicating 'split') ---
# edges_left  = np.linspace(left_min,  left_max,  n_left + 1)
# edges_right = np.linspace(right_min, right_max, n_right + 1)[1:]
# edges = np.r_[edges_left, edges_right]  # strictly increasing edges
#
# xc, yc = conformist_contrarian_coupling_array_connected.ravel(), mean_module_zeta_array_connected.ravel()
# y_meanc, edgesc, _ = binned_statistic(xc, yc, statistic=lambda v: np.nanmean(v) if v.size > 0 else np.nan, bins=edges)
# x_midc = 0.5 * (edgesc[:-1] + edgesc[1:])
# # y_stdc  = binned_statistic(x, y, statistic=lambda v: np.nanstd(v, ddof=0) if v.size > 0 else np.nan, bins=edgesc)[0]
# y_loc = binned_statistic(xc, yc, statistic=lambda v: np.nanpercentile(v, 16), bins=edges)[0]
# y_hic = binned_statistic(xc, yc, statistic=lambda v: np.nanpercentile(v, 84), bins=edges)[0]
# countsc = binned_statistic(xc, yc, statistic=lambda v: np.sum(~np.isnan(v)), bins=edgesc)[0]
# # maskc = countsc >= 1 # Keep bins with data; set std=0 (or np.nan) where <2 points
# # y_stdc[countsc < 2] = np.nan  # or np.nan if you prefer gaps
# # x_plotc = x_midc[maskc]
# # y_mean_plotc = y_meanc[maskc]
# # y_std_plotc = y_stdc[maskc]
# # ax.plot(x_plotc, y_mean_plotc, lw=2) # , color="#fde0d0")
# # ax.fill_between(x_plotc, y_mean_plotc - y_std_plotc, y_mean_plotc + y_std_plotc, alpha=0.2)
# ax.plot(x_midc, y_meanc, lw=lw, label="$w_s = 5$") # , color="#fde0d0")
# ax.fill_between(x_midc, y_loc, y_hic, alpha=0.2)
# # ax.axvline(x=0.03, linestyle='--', color=deep[1])


""" source weight = 20 """
# --- choose your split and bin counts ---
split = 0
left_min, left_max, n_left  = -0.6,  split, 20   # small number of bins on the left
right_min, right_max, n_right = split, 0.6, 200  # small number of bins on the right

# --- build edges (note [1:] to avoid duplicating 'split') ---
edges_left  = np.linspace(left_min,  left_max,  n_left + 1)
edges_right = np.linspace(right_min, right_max, n_right + 1)[1:]
edges = np.r_[edges_left, edges_right]  # strictly increasing edges

xc2, yc2 = conformist_contrarian_coupling_array_connected2.ravel(), mean_module_zeta_array_connected2.ravel()
y_meanc2, edgesc2, _ = binned_statistic(xc2, yc2, statistic=lambda v: np.nanmean(v) if v.size > 0 else np.nan, bins=edges)
x_midc2 = 0.5 * (edgesc2[:-1] + edgesc2[1:])
y_loc2 = binned_statistic(xc2, yc2, statistic=lambda v: np.nanpercentile(v, 16), bins=edges)[0]
y_hic2 = binned_statistic(xc2, yc2, statistic=lambda v: np.nanpercentile(v, 84), bins=edges)[0]
countsc = binned_statistic(xc2, yc2, statistic=lambda v: np.sum(~np.isnan(v)), bins=edgesc2)[0]
ax.plot(x_midc2, y_meanc2, lw=lw, label="$\mathcal{A}_s \\neq 0$", color="#fde0d0")
ax.fill_between(x_midc2, y_loc2, y_hic2, alpha=0.2, color="#fde0d0")

ax.set_xlabel("$\\gamma$") # $\sum_{k\in\mathcal{W}_7}W_{jk}cos(alpha_{jk})$
ax.set_ylabel("$\\langle |Z e^{-i\phi}| \\rangle_t$")
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.01, 1.05])
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([0, 0.5, 1])

ax.scatter(-0.3492930618693419, 0.3435265384981321, color="#aac4ff", marker="s", zorder=10)
ax.scatter(-0.3492930618693419, 0.6178451580598752, color="#fde0d0", marker="s", zorder=10)
ax.scatter(-0.11853950874113284, 0.7777158044773287, color="#aac4ff", marker="*",  zorder=10)
ax.scatter(-0.11853950874113284, 0.9999999999999996, color="#fde0d0", marker="*",  zorder=10)

mygrey = "#666666"
ax.tick_params(colors=mygrey)          # tick marks and numbers
for spine in ax.spines.values():
    spine.set_color(mygrey)

ax.xaxis.label.set_color(mygrey)
ax.yaxis.label.set_color(mygrey)
ax.title.set_color(mygrey)

ax.legend(loc=4, labelcolor="#6784c7", handlelength=1)

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