# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import json
from plots.config_rcparams import *
from pathlib import Path
import time
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog
from matplotlib.patches import Polygon


""" The data are generated from synchronization_cross_ratio_part_oneinstance.py """
def get_param_dictionary(filepath):
    filepath = Path(filepath)
    with filepath.open("r") as f:
        params = json.load(f)
    return params
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
path = REPO_ROOT / 'simulations' / 'kooku1_fig3_data'


t0, t1, dt = 0, 300, 0.01
timelist = np.linspace(t0, t1, int(t1 / dt))
percentage_averaged_end_time_series = 0.5
start_idx = int(percentage_averaged_end_time_series*len(timelist))

""" Freq sync to periodic state """
file_path_freqsync1 = Path(path / "2025_08_21_15h06min08sec_freqsync_15h03_kuramoto_parameters_dictionary.json")
dict_freqsync1 = get_param_dictionary(file_path_freqsync1)
conformist_contrarian_coupling_freqsync1 = dict_freqsync1['conformist_contrarian_coupling']
print("(isolated) conformist_contrarian_couplingm_freqsync1 = ", conformist_contrarian_coupling_freqsync1)
alpha = np.array(dict_freqsync1['alpha'])
print("alpha_s_freqsync1 = ", alpha[7, 0])
Rezeta_freqsync1 = np.array(dict_freqsync1["Rezeta"])
Imzeta_freqsync1 = np.array(dict_freqsync1["Imzeta"])
zeta_freqsync1 = Rezeta_freqsync1 + 1j*Imzeta_freqsync1
print(np.mean(np.abs(zeta_freqsync1[start_idx:])))

file_path_periodic = Path(path / "2025_08_21_15h03min45sec_periodic_kuramoto_parameters_dictionary.json")
dict_periodic = get_param_dictionary(file_path_periodic)
conformist_contrarian_coupling_periodic = dict_periodic['conformist_contrarian_coupling']
print("conformist_contrarian_coupling_periodic = ", conformist_contrarian_coupling_periodic)
Rezeta_periodic = np.array(dict_periodic["Rezeta"])
Imzeta_periodic = np.array(dict_periodic["Imzeta"])
zeta_periodic = Rezeta_periodic + 1j*Imzeta_periodic
print(np.mean(np.abs(zeta_periodic[start_idx:])))
alpha = np.array(dict_periodic['alpha'])
print("alpha_s_periodic = ", alpha[7, 0])


""" Freq sync to phase sync """
file_path_freqsync2 = Path(path / "2025_08_21_15h32min23sec_freq_sync_15h29_kuramoto_parameters_dictionary.json")
dict_freqsync2 = get_param_dictionary(file_path_freqsync2)
conformist_contrarian_coupling_freqsync2 = dict_freqsync2['conformist_contrarian_coupling']
print("(isolated) conformist_contrarian_coupling_freqsync2 = ", conformist_contrarian_coupling_freqsync2)
Rezeta_freqsync2 = np.array(dict_freqsync2["Rezeta"])
Imzeta_freqsync2 = np.array(dict_freqsync2["Imzeta"])
zeta_freqsync2 = Rezeta_freqsync2 + 1j*Imzeta_freqsync2
print(np.mean(np.abs(zeta_freqsync2[start_idx:])))
alpha = np.array(dict_freqsync2['alpha'])
print("alpha_s_freqsync2 = ", alpha[7, 0])


file_path_phasesync = Path(path / "2025_08_21_15h29min15sec_phasesync_kuramoto_parameters_dictionary.json")
dict_phasesync = get_param_dictionary(file_path_phasesync)
conformist_contrarian_coupling_phasesync = dict_phasesync['conformist_contrarian_coupling']
print("conformist_contrarian_coupling_phasesync = ", conformist_contrarian_coupling_phasesync)
Rezeta_phasesync = np.array(dict_phasesync["Rezeta"])
Imzeta_phasesync = np.array(dict_phasesync["Imzeta"])
zeta_phasesync = Rezeta_phasesync + 1j*Imzeta_phasesync
print(np.mean(np.abs(zeta_phasesync[start_idx:])))
alpha = np.array(dict_phasesync['alpha'])
print("alpha_s_phasesync = ", alpha[7, 0])



""" Plot result """

def add_2d_arrowhead(ax, tip, direction, width=0.2, height=0.4, color='k', edgecolor='none', zorder=10, **patch_kw):
    """
    Draws a triangular arrowhead at the given tip, pointing along 'direction'.

    Parameters:
        ax        : matplotlib.axes.Axes
        tip       : (x, y) tip of the arrow (on the curve)
        direction : (dx, dy) vector indicating arrow direction (will be normalized)
        width     : width of the base (data units)
        height    : distance from tip to base (arrow length, data units)
        color     : triangle face color
        edgecolor : triangle edge color (default none)
        zorder    : drawing order
        **patch_kw: forwarded to matplotlib.patches.Polygon
    """
    tip = np.asarray(tip, dtype=float)
    direction = np.asarray(direction, dtype=float)
    L = np.linalg.norm(direction)
    if not np.isfinite(L) or L == 0:
        return None
    d = direction / L

    # Base center behind the tip
    base_center = tip - height * d

    # 2D perpendicular to d (rotate by +90°)
    ortho = np.array([-d[1], d[0]])
    ortho /= np.linalg.norm(ortho)
    ortho *= (width / 2.0)

    base1 = base_center + ortho
    base2 = base_center - ortho

    triangle = np.vstack([base1, base2, tip])
    poly = Polygon(triangle, closed=True, facecolor=color, edgecolor=edgecolor, zorder=zorder, **patch_kw)
    ax.add_patch(poly)
    return poly

plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})
s = 15
width, height = 0.1, 0.1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2))
ax1.set_aspect('equal')
angle = np.linspace(0, 2*np.pi, 300)

ax1.plot(np.cos(angle), np.sin(angle), color=reduced_grey, alpha=0.8, zorder=-10)
ax1.plot(Rezeta_freqsync1, Imzeta_freqsync1, color="#aac4ff", label=r"Isolated")
mid = 5  # int(len(Rezeta_freqsync1) // 5)
p_tip = [Rezeta_freqsync1[mid], Imzeta_freqsync1[mid]]
vec   = [Rezeta_freqsync1[mid] - Rezeta_freqsync1[mid-1],
         Imzeta_freqsync1[mid] - Imzeta_freqsync1[mid-1]]
add_2d_arrowhead(ax1, tip=p_tip, direction=vec, width=width, height=height, color="#aac4ff", zorder=15)

ax1.plot(Rezeta_periodic, Imzeta_periodic, color="#fde0d0", label=r"Pertubed")
mid = 10
p_tip = [Rezeta_periodic[mid], Imzeta_periodic[mid]]
vec   = [Rezeta_periodic[mid] - Rezeta_periodic[mid-1],
         Imzeta_periodic[mid] - Imzeta_periodic[mid-1]]
add_2d_arrowhead(ax1, tip=p_tip, direction=vec, width=width, height=height, color="#fde0d0", zorder=15)

ax1.scatter(Rezeta_periodic[0], Imzeta_periodic[0], color=dark_grey, s=s)
ax1.scatter(Rezeta_freqsync1[-1], Imzeta_freqsync1[-1], color="#aac4ff", s=s, marker="s")
ax1.axis('off')
# ax1.legend(loc=1)

ax2.set_aspect('equal')
ax2.plot(np.cos(angle), np.sin(angle), color=reduced_grey, alpha=0.8, zorder=-10)

ax2.plot(Rezeta_freqsync2, Imzeta_freqsync2, color="#aac4ff", label=r"Isolated")
mid = 20  # int(len(Rezeta_freqsync1) // 5)
p_tip = [Rezeta_freqsync2[mid], Imzeta_freqsync2[mid]]
vec   = [Rezeta_freqsync2[mid] - Rezeta_freqsync2[mid-1],
         Imzeta_freqsync2[mid] - Imzeta_freqsync2[mid-1]]
add_2d_arrowhead(ax2, tip=p_tip, direction=vec, width=width, height=height, color="#aac4ff", zorder=15)


ax2.plot(Rezeta_phasesync, Imzeta_phasesync, color="#fde0d0", label=r"Pertubed")
mid = 10
p_tip = [Rezeta_phasesync[mid], Imzeta_phasesync[mid]]
vec   = [Rezeta_phasesync[mid] - Rezeta_phasesync[mid-1],
         Imzeta_phasesync[mid] - Imzeta_phasesync[mid-1]]
add_2d_arrowhead(ax2, tip=p_tip, direction=vec, width=width, height=height, color="#fde0d0", zorder=15)


ax2.scatter(Rezeta_phasesync[0], Imzeta_phasesync[0], color=dark_grey, s=s)
ax2.scatter(Rezeta_phasesync[-1], Imzeta_phasesync[-1], color="#fde0d0", s=40, marker="*")
ax2.scatter(Rezeta_freqsync2[-1], Imzeta_freqsync2[-1], color="#aac4ff", s=40, marker="*")
ax2.axis('off')
# ax2.legend(loc=1)

plt.show()
app = QApplication(sys.argv)
reply = QMessageBox.question(None, "Python", "Would you like to save the parameters, the data, and the plot?",
                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
if reply == QMessageBox.StandardButton.Yes:
    filename, ok = QInputDialog.getText(None, "File:", "Enter your file name")
    if ok:
        print("File name:", filename)
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    fig.savefig(path / f'{timestr}_{filename}_synchro_crossratio_part_kuramoto.pdf')
    fig.savefig(path / f'{timestr}_{filename}_synchro_crossratio_part_kuramoto.png')
