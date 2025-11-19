#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pathlib import Path
from plots.config_rcparams import *
from math import floor, log10

pie_chart = False


def sig_figs(x: float, precision: int):
    """ https://mattgosden.medium.com/rounding-to-significant-figures-in-python-2415661b94c3
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))


def fix_xticks_replace_zero(ax, data, bins=50):
    """
    Replace the x-tick labeled '0' with the center of the first histogram bin.
    """
    # compute bins from the data
    counts, bin_edges = np.histogram(data, bins=bins)

    # bin centers
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    first_center = sig_figs(centers[0], 1)

    # get current ticks
    xticks = ax.get_xticks()
    xlabels = []

    for t in xticks:
        if abs(t) < 1e-12:      # detect tick == 0
            xlabels.append(f"{first_center:.3g}")
        else:
            xlabels.append(f"{t:.3g}")

    ax.set_xticklabels(xlabels)

# --- Load aligned text file using fixed-width parser ---
ROOT = Path(__file__).resolve().parents[1]
in_file = ROOT / "simulations" / "kooku1_fig4_data" / "networks_constants_motion.txt"

# Read fixed-width table (header at line 0, dashed line at line 1)
df = pd.read_fwf(in_file, header=0, skiprows=[1])

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# --- Parse column types ---

# isdirected is stored as text in the txt; convert to bool
df["isdirected"] = (df["isdirected"].astype(str).str.strip().map({"True": True, "False": False}))

# Basic numeric columns
df["nb_cte_motion"]           = df["nb_cte_motion"].astype(float)
df["nb_vertices"]             = df["nb_vertices"].astype(float)
df["nb_sources"]              = df["nb_sources"].astype(float)
df["nb_2sources"]             = df["nb_2sources"].astype(float)
df["nb_conserved_crossratio"] = df["nb_conserved_crossratio"].astype(float)

# --------------------------
# Compute summary statistics
# -------------------------
number_of_networks_dataset = len(df)

directed_networks   = df[df["isdirected"]]
undirected_networks = df[~df["isdirected"]]

number_directed   = len(directed_networks)
number_undirected = len(undirected_networks)

connectomes       = df[df["type"] == "connec"]
powergrids        = df[df["type"] == "powerg"]
social_networks   = df[df["type"] == "social"]

number_connectomes = len(connectomes)
number_powergrids  = len(powergrids)
number_social      = len(social_networks)

zero_cte = df[df["nb_cte_motion"] == 0]
zero_cte_undir = zero_cte[~zero_cte["isdirected"]]
zero_cte_dir   = zero_cte[zero_cte["isdirected"]]
number_zero_cte = len(zero_cte)

zero_source = df[df["nb_sources"] == 0]
number_zero_source = len(zero_source)

zero_2source = df[df["nb_2sources"] == 0]
number_zero_2source = len(zero_2source)

zero_cross = df[df["nb_conserved_crossratio"] == 0]
number_zero_cross = len(zero_cross)

# Helper for percentages
pct = lambda x: f"{100*x/number_of_networks_dataset:.1f}\%"
pctdir = lambda x: f"$^*${100*x/number_directed:.1f}\%"

# Create a DataFrame for the table
table_df = pd.DataFrame([
    ["Empirical networks", number_of_networks_dataset, ""],
    ["Directed networks", number_directed, pct(number_directed)],
    ["Undirected networks", number_undirected, pct(number_undirected)],
    ["Social networks", number_social, pct(number_social)],
    ["Power grids", number_powergrids, pct(number_powergrids)],
    ["Connectomes", number_connectomes, pct(number_connectomes)],
    ["Networks admitting\nconstants of motion", number_of_networks_dataset - number_zero_cte,
     pct(number_of_networks_dataset - number_zero_cte)],
    ["Networks with sources", number_of_networks_dataset - number_zero_source,
     pctdir(number_of_networks_dataset - number_zero_source)],
    ["Networks with 2-sources", number_of_networks_dataset - number_zero_2source,
     pctdir(number_of_networks_dataset - number_zero_2source)],
    ["Networks admitting\nconserved cross-ratios", number_of_networks_dataset - number_zero_cross,
     pct(number_of_networks_dataset - number_zero_cross)],
    # [" --- Directed", len(zero_cte_dir), pct(len(zero_cte_dir))],
    # [" --- Undirected", len(zero_cte_undir), pct(len(zero_cte_undir))],
], columns=["Category", "Count", "\%"])


# --- Adjust number of conserved cross-ratios to remove the sources ---
isdir   = df["isdirected"]
sources = df["nb_sources"]
cr_orig = df["nb_conserved_crossratio"]

cr_nosource = np.where(isdir & (sources >= 4), cr_orig - (sources - 3), cr_orig)

df["nb_conserved_crossratio_nosource"] = cr_nosource

# --- Compute ratios per vertex ---

df["total_ratio"]  = df["nb_cte_motion"] / df["nb_vertices"]
df["1source_ratio"] = df["nb_sources"] / df["nb_vertices"]
df["2source_ratio"] = df["nb_2sources"] / df["nb_vertices"]
df["cr_ratio"]      = df["nb_conserved_crossratio_nosource"] / df["nb_vertices"]


def nz(x):
    """Return only nonzero values of a Series."""
    return x[x > 0]


# Split into undirected and directed networks
df_undirected = df[~df["isdirected"]]
df_directed   = df[df["isdirected"]]




""" -----------------------  Plot Table ------------------------------ """
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")
table = ax.table(
    cellText     = table_df.values,
    colLabels    = table_df.columns,
    cellLoc      = "center",
    loc          = "center")

# Styling
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.3)

# remove lines in table
for key, cell in table.get_celld().items():
    cell.set_edgecolor("none")

# Bold header
for key, cell in table.get_celld().items():
    if key[0] == 0:   # header row
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#f2f2f2")

# Set column width
for row in range(len(table_df) + 1):
    table[(row, 1)].set_width(0.08)  # Count column
    table[(row, 2)].set_width(0.10)  # Percent column
    table[(row, 0)].set_width(0.3)  # Category column
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12,
})

# fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=False)
# (ax1, ax2), (ax3, ax4) = axes

fig = plt.figure(figsize=(6, 3.5))
gs = GridSpec(nrows=3, ncols=2 , width_ratios=[1, 1], height_ratios=[1, 1, 1])
# ax = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])


bins = 40

# 1) Lower-bound on the fraction nb_cte_motion / nb_vertices
weights1 = np.ones_like(nz(df_directed["total_ratio"])) / len(nz(df_directed["total_ratio"]))
weights11 = np.ones_like(nz(df_undirected["total_ratio"])) / len(nz(df_undirected["total_ratio"]))
counts, bins, _ = ax1.hist(nz(df_directed["total_ratio"]),   bins=bins, alpha=0.7, label="Directed networks", weights=weights1, color="#333333")
ax1.hist(nz(df_undirected["total_ratio"]), bins=bins, alpha=0.8, label="Undirected networks\n(\# cross-ratio)",  weights=weights11, color="#d1d1d1")
ax1.set_xlabel("Lower bound on \# constants of motion / $N$")
ax1.set_ylabel("Normalized frequency")
ax1.set_yticks([0, 0.1, 0.2])
ax1.set_ylim([-0.002, 0.201])
ax1.legend()
fix_xticks_replace_zero(ax1, nz(df["total_ratio"]), bins=50)

if pie_chart:
    counts = {
        "Social$\\quad$\nnetworks" : number_social,
        "Connectomes" : number_connectomes,
        "Powergrids" : number_powergrids
    }
    colors = ["#99c3d1", "#f2c983", "#cbdeaf"]
    total = sum(counts.values())
    pct = {k: 100*v/total for k, v in counts.items()}
    labels = [f"\n{k}\n{pct[k]:.0f}\%" for k in counts]
    inset = ax1.inset_axes([0.5, 0.3, 0.35, 0.45])   # (x0, y0, width, height)
    inset.pie(counts.values(), labels=labels, labeldistance=1.15,  # {k}\n{v}
              textprops={"fontsize": 6}, colors=colors)  #  autopct=lambda p: f"{p:.0f}%"
    # inset.set_title("Dataset composition", fontsize=8)
    inset.title.set_position((0.5, 0.5))


# 2) Number of conserved cross-ratios (not sources_
weights2 = np.ones_like(nz(df_directed["cr_ratio"])) / len(nz(df_directed["cr_ratio"]))
ax2.hist(nz(df_directed["cr_ratio"]), bins=bins, alpha=0.7, label="Directed", weights=weights2, color="#333333")
# ax2.set_xlabel(r"\# cross-ratios / $N$") # (no sources)")
# ax2.set_ylabel("Normalized frequency")
# ax2.legend()
ax2.set_ylim([-0.02, 0.351])
ax2.set_xticks([0.01, 0.8])
fix_xticks_replace_zero(ax2, nz(df_directed["cr_ratio"]), bins=50)


# 3) Number of sources (here you plotted directed only; keep that choice)
weights3 = np.ones_like(nz(df_directed["1source_ratio"])) / len(nz(df_directed["1source_ratio"]))
ax3.hist(nz(df_directed["1source_ratio"]), bins=bins, alpha=0.7, label="Directed", weights=weights3, color="#333333")
# ax3.set_xlabel(r"\# sources / $N$")
# ax3.set_ylabel("Normalized frequency")
# ax3.legend()
ax3.set_ylim([-0.005, 0.09])
ax3.set_xticks([0.01, 0.8])
fix_xticks_replace_zero(ax3, nz(df_directed["1source_ratio"]), bins=bins)


# 4) Number of 2-sources (again directed only)
weights4 = np.ones_like(nz(df_directed["2source_ratio"])) / len(nz(df_directed["2source_ratio"]))
ax4.hist(nz(df_directed["2source_ratio"]), bins=50, alpha=0.7, label="Directed", weights=weights4, color="#333333")
# ax4.set_xlabel(r"\# 2-sources / $N$")
# ax4.set_ylabel("Normalized frequency")
# ax4.legend()
ax4.set_ylim([-0.01, 0.15])

fix_xticks_replace_zero(ax4, nz(df_directed["2source_ratio"]), bins=50)

fig.tight_layout()

out = ROOT / "plots" / "cte_motion_histograms.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out)
plt.show()

print(f"Saved histogram figure → {out}")
















#
# # number_of_networks_dataset = len(df)
# # print("Number of networks in dataset:", number_of_networks_dataset)
# #
# # directed_networks = df[df["isdirected"] == True]
# # number_of_directed_networks = len(directed_networks)
# # print("Number of directed networks in dataset:", number_of_directed_networks)
# # print("Number of undirected networks in dataset:", number_of_networks_dataset - number_of_directed_networks)
# #
# # # print(df[["name", "type"]][200:250])
# # connectomes = df[df["type"] == "connec"]
# # powergrids = df[df["type"] == "powerg"]
# # social_networks = df[df["type"] == "social"]
# # number_connectomes = len(connectomes)
# # number_powergrids = len(powergrids)
# # number_social_networks = len(social_networks)
# # print("\nNumber of connectomes in dataset:", number_connectomes,
# #      f"(~{sig_figs(number_connectomes/number_of_networks_dataset*100, 2)}%)")
# # print("Number of powergrids in dataset:", number_powergrids,
# #      f"(~{sig_figs(number_powergrids/number_of_networks_dataset*100, 2)}%)")
# # print("Number of social networks in dataset:", number_social_networks,
# #      f"(~{sig_figs(number_social_networks/number_of_networks_dataset*100, 2)}%)")
# #
# # zero_cte = df[df["nb_cte_motion"] == 0]
# # count_zero = len(zero_cte)
# # print("\nNumber of networks with 0 constants of motion:", count_zero)
# #
# # zero_cte_undirected = zero_cte[~zero_cte["isdirected"]]
# # count_zero_undirected = len(zero_cte_undirected)
# # print("Number of directed networks with 0 constants of motion:", count_zero - count_zero_undirected)
# # print("Number of undirected networks with 0 constants of motion:", count_zero_undirected)
# #
# # zero_cte = df[df["nb_conserved_crossratio"] == 0]
# # count_zero = len(zero_cte)
# # print("\nNumber of networks with 0 cross-ratio:", count_zero)
# #
#
#
#
#
# # print("\nNetworks with 0 constants of motion (with directedness):")
# # for _, row in zero_cte.iterrows():
# #     name = row["name"]
# #     isdir = row["isdirected"]
# #     kind = "directed" if isdir else "undirected"
# #     print(f"  - {name:40s}   ({kind})")
#