import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import json
from pathlib import Path
from plots.config_rcparams import dark_grey
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_parameters_to_globals(filepath, keys_as_array=None):
    """
    Load a parameter dictionary from JSON and unpack into globals(). See symmetry_changes_invariant_sets.py or the keys
    of the parameter dictionary to see all the parameters.
    Converts specified list-valued keys back into NumPy arrays.

    Parameters:
        filepath (str or Path): path to the JSON file
        keys_as_array (list): keys that should be converted back to np.array
    """
    filepath = Path(filepath)
    with filepath.open("r") as f:
        params = json.load(f)

    keys_as_array = set(keys_as_array or [])
    for key, val in params.items():
        if key in keys_as_array and isinstance(val, list):
            val = np.array(val)
        globals()[key] = val
SCRIPT_DIR = Path(__file__).resolve().parent  # Get current script location
REPO_ROOT = SCRIPT_DIR.parent  # Go to repo root (adjust this based on how deep your script is)
path = REPO_ROOT / 'simulations' / 'kooku1_fig3_data'  # Path to the data file
# Path to your JSON file
file_name = "2025_08_04_13h15min51sec_higher_value_cross_ratio_kuramoto_parameters_dictionary.json"
file_path = Path(path / file_name)
load_parameters_to_globals(file_path, keys_as_array=[
    "sizes_monomial", "sizes_crossratio", "size_nonintegrable",
    "W", "alpha", "omega", "C", "chi", "theta0", "theta",
    "epsilon_array", "random_exponents",
    "probabilities_monomial", "probabilities_crossratio",
    "probabilities_nonintegrable2", "probabilities_monomial2", "probabilities_crossratio2", "timelist"])

# plt.plot(theta[:, :5])
# plt.show()


def draw_axes_at_origin_with_pi_ticks(ax, axis_label, axis_length=2*np.pi, tick_spacing=np.pi, offset=0.1, fontsize=10):
    """
    Draws custom 3D axes at (0, 0, 0) with π-based ticks and LaTeX labels.
    Hides the default 3D box. Works on any Axes3D object.
    """
    # 1. Hide default cube and ticks
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax._axis3don = False  # hide all 3D spines/ticks

    # 2. Draw custom axis lines
    ax.plot([0, axis_length], [0, 0], [0, 0], color=dark_grey, lw=1)  # x-axis
    ax.plot([0, 0], [0, axis_length], [0, 0], color=dark_grey, lw=1)  # y-axis
    ax.plot([0, 0], [0, 0], [0, axis_length], color=dark_grey, lw=1, zorder=100)  # z-axis in front

    # 3. Define π ticks and labels
    ticks = np.arange(0, axis_length + 1e-3, tick_spacing)
    tick_labels = []
    for t in ticks:
        if np.isclose(t, 0):
            tick_labels.append("0")
        elif np.isclose(t, np.pi):
            tick_labels.append(r"$\pi$")
        elif np.isclose(t, 2*np.pi):
            tick_labels.append(r"$2\pi$")
        elif (t / np.pi).is_integer():
            tick_labels.append(fr"${int(t / np.pi)}\pi$")
        else:
            tick_labels.append(fr"${t / np.pi:.1g}\pi$")

    # 4. Draw ticks and labels on each axis
    for t, label in zip(ticks, tick_labels):
        # X axis
        ax.plot([t, t], [-0.8*offset, 0], [0, 0], color=dark_grey, lw=1)
        ax.text(t - 0.4, -4*offset, 0, label, ha='center', va='top', fontsize=fontsize)

        if label != "0":
            # Y axis
            ax.plot([-0.8*offset, 0], [t, t], [0, 0], color=dark_grey, lw=1)
            ax.text(-3*offset, t, 0, label, ha='right', va='center', fontsize=fontsize)


            # Z axis
            ax.plot([0, 0], [-0.8*offset, 0], [t, t], color=dark_grey, lw=1, zorder=100)
            ax.text(0, -4.5*offset, t, label, ha='left', va='center', fontsize=fontsize)

    # 5. Axis labels (LaTeX style)
    ax.text(axis_length + 1, 0, 0, axis_label[0], fontsize=fontsize + 2)
    ax.text(0.1, axis_length + 0.7, 0, axis_label[1], fontsize=fontsize + 2)
    ax.text(0, -offset, axis_length + 0.5, axis_label[2], fontsize=fontsize + 2)

    # 6. Set limits
    ax.set_xlim(0, axis_length)
    ax.set_ylim(0, axis_length)
    ax.set_zlim(0, axis_length)


def add_3d_arrowhead(ax, tip, direction, width=0.2, height=0.4, color='k'):
    """
    Draws a triangular arrowhead at the given tip, pointing along 'direction'.

    Parameters:
        ax        : Axes3D object
        tip       : (x, y, z) tip of the arrow (on the curve)
        direction : vector indicating arrow direction (will be normalized)
        width     : width of the base
        height    : distance from tip to base (arrow length)
        color     : triangle color
    """
    tip = np.array(tip, dtype=float)
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    # Define triangle base center (behind the tip)
    base_center = tip - height * direction

    # Construct orthogonal vector in the triangle plane
    ref = np.array([0, 0, 1]) if abs(direction[2]) < 0.9 else np.array([1, 0, 0])
    ortho = np.cross(direction, ref)
    ortho /= np.linalg.norm(ortho)
    ortho *= width / 2

    base1 = base_center + ortho
    base2 = base_center - ortho

    triangle = [base1, base2, tip]

    arrowhead = Poly3DCollection([triangle], color=color, edgecolor='none')
    ax.add_collection3d(arrowhead)


def plot_invariant_set_cross_ratio_c1234(ax, c, n, epsilon, cut_value=0.05, color='blue', lw=0.5, alpha=0.9, XgeqY=True):

    # Domain [0, 2π]
    x = np.linspace(0 + cut_value, 2 * np.pi - cut_value, n)
    y = np.linspace(0 + cut_value, 2 * np.pi - cut_value, n)
    z = np.linspace(0 + cut_value, 2 * np.pi - cut_value, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Shape: (n, n, n)

    # Compute function F, theta1 = 0
    numerator = np.sin(Y / 2) * np.sin((Z - X) / 2)
    denominator = np.sin((Y - X) / 2) * np.sin(Z / 2)

    # Mask singularities
    mask_singularity = np.abs(denominator) < epsilon

    if XgeqY:
        # Mask out where theta_2 < theta_3
        mask_theta2_geq_theta3 = X <= Y
        full_mask = mask_singularity | mask_theta2_geq_theta3
    else:
        # Mask out where theta_2 < theta_3
        mask_theta2_leq_theta3 = Y <= X
        full_mask = mask_singularity | mask_theta2_leq_theta3

    denominator[full_mask] = np.nan

    # Compute and mask F
    F = numerator / denominator

    # Define level set
    F_level = F - c

    verts, faces, normals, values = measure.marching_cubes(F_level, level=0.0)

    # Rescale verts to physical coordinates [0, 2π]
    scale = (2 * np.pi) / (n - 1)
    verts_scaled = verts * scale

    mesh = ax.plot_trisurf(verts_scaled[:, 0], verts_scaled[:, 1], faces, verts_scaled[:, 2],
                           color=color, lw=lw, alpha=alpha, shade=False)  # cmap='viridis',


def plot_invariant_set_sym_gen(ax, omega, omega1, s, phaselag, c, n, epsilon, cut_value=0.05, color="blue", lw=0.5, alpha=0.9):

    # Domain [0, 2π]
    x = np.linspace(0 + cut_value, 2 * np.pi - cut_value, n)
    y = np.linspace(0 + cut_value, 2 * np.pi - cut_value, n)
    z = np.linspace(0 + cut_value, 2 * np.pi - cut_value, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Shape: (n, n, n)


    # Compute function F
    numerator = (omega - omega1 + s*np.sin(-X - phaselag))*np.sin((Y - Z)/2)
    denominator = np.sin((Y - X) / 2) * np.sin((Z- X) / 2)

    # Mask singularities
    mask_singularity = np.abs(denominator) < epsilon

    # Combine masks
    full_mask = mask_singularity
    denominator[full_mask] = np.nan

    # Compute and mask F
    F = numerator / denominator

    # Define level set
    F_level = F - c

    verts, faces, normals, values = measure.marching_cubes(F_level, level=0.0)

    # Rescale verts to physical coordinates [0, 2π]
    scale = (2 * np.pi) / (n - 1)
    verts_scaled = verts * scale

    mesh = ax.plot_trisurf(verts_scaled[:, 0], verts_scaled[:, 1], faces, verts_scaled[:, 2],
                           color=color, lw=lw, alpha=alpha, shade=False)



""" Get trajectories """
theta2 = theta[:, 1]
theta345_transformed = np.array(theta345_transformed)
theta345_transformed = np.where(theta345_transformed < 0, 2*np.pi + theta345_transformed, theta345_transformed)

epsilon_index1 = 0  # epsilon = 0
theta31 = np.array(theta345_transformed)[epsilon_index1, :, 0]
theta41 = np.array(theta345_transformed)[epsilon_index1, :, 1]
theta51 = np.array(theta345_transformed)[epsilon_index1, :, 2]

epsilon_index2 = -1  # epsilon in 1,2,3,...,-1
theta32 = np.array(theta345_transformed)[epsilon_index2, :, 0]
theta42 = np.array(theta345_transformed)[epsilon_index2, :, 1]
theta52 = np.array(theta345_transformed)[epsilon_index2, :, 2]

""" Plot invariant sets and trajectories """

n = 100
alphaplot = 0.75
linewidth = 1.8
offset1 = 0.3
offset2 = 0.3
plt.rcParams.update({
    "text.usetex": True,               # Use LaTeX for all text
    "font.family": "serif",            # Use serif fonts
    "font.serif": ["Computer Modern"], # Same as LaTeX default
    "axes.unicode_minus": False        # Allow proper minus signs
})



fig = plt.figure(figsize=(4, 5))

cut_value = 0.15
mid = len(theta2) // 6

ax = fig.add_subplot(211, projection='3d')
# ax.scatter(theta2[0], theta31[0], theta41[0], color="k", s=10)
# ax.scatter(theta2[0], theta32[0], theta42[0], color='#2f6eff', s=10)
ax.plot(theta2, theta31, theta41, color="k", linewidth=linewidth)
p_tip = [theta2[mid], theta31[mid], theta41[mid]]
vec = np.array([theta2[mid] - theta2[mid - 1],
                theta31[mid] - theta31[mid - 1],
                theta41[mid] - theta41[mid - 1]])
add_3d_arrowhead(ax, tip=p_tip, direction=vec, width=0.4, height=0.5, color="k")
ax.plot(theta2, theta32, theta42, color='#2f6eff', linewidth=linewidth)
p_tip = [theta2[mid], theta32[mid], theta42[mid]]
vec = np.array([theta2[mid] - theta2[mid - 1],
                theta32[mid] - theta32[mid - 1],
                theta42[mid] - theta42[mid - 1]])
add_3d_arrowhead(ax, tip=p_tip, direction=vec, width=0.4, height=0.5, color="#2f6eff")
plot_invariant_set_cross_ratio_c1234(ax, cross_ratio_vs_epsilon[epsilon_index1], n,  color='#6784c7',
                                     cut_value=cut_value, epsilon=1e-6, lw=0.5, alpha=alphaplot)
plot_invariant_set_cross_ratio_c1234(ax, cross_ratio_vs_epsilon[epsilon_index2], n,  color='#aac4ff',
                                     cut_value=cut_value, epsilon=1e-6, lw=0.5, alpha=alphaplot-0.2)
plot_invariant_set_cross_ratio_c1234(ax, cross_ratio_vs_epsilon[epsilon_index2], n,  color='#aac4ff',
                                     cut_value=cut_value, epsilon=1e-6, lw=0.5, alpha=alphaplot-0.2, XgeqY=False)
plot_invariant_set_cross_ratio_c1234(ax, cross_ratio_vs_epsilon[epsilon_index1], n,  color='#6784c7',
                                     cut_value=cut_value, epsilon=1e-6, lw=0.5, alpha=alphaplot, XgeqY=False)
print(cross_ratio_vs_epsilon[epsilon_index1], cross_ratio_vs_epsilon[epsilon_index2])
# ax.set_title(r'Invariant sets of $c_{1234}$')
ax.grid(False)
# ax.view_init(elev=27, azim=42)
ax.view_init(elev=35, azim=53)
ax_labels = [r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
draw_axes_at_origin_with_pi_ticks(ax, ax_labels, axis_length=2*np.pi, tick_spacing=np.pi, offset=offset1, fontsize=10)


cut_value = 0

ax2 = fig.add_subplot(212, projection='3d')
# ax2.scatter(theta2[0], theta41[0], theta51[0], color="k", s=10)
# ax2.scatter(theta2[0], theta42[0], theta52[0], color='#6cca31', s=10)
ax2.plot(theta2, theta41, theta51, linewidth=linewidth, color="k")
p_tip = [theta2[mid], theta41[mid], theta51[mid]]
vec = np.array([theta2[mid] - theta2[mid - 1],
                theta41[mid] - theta41[mid - 1],
                theta51[mid] - theta51[mid - 1]])
add_3d_arrowhead(ax2, tip=p_tip, direction=vec, width=0.4, height=0.5, color="k")
ax2.plot(theta2, theta42, theta52, linewidth=linewidth, color='#6cca31')
p_tip = [theta2[mid], theta42[mid], theta52[mid]]
vec = np.array([theta2[mid] - theta2[mid - 1],
                theta42[mid] - theta42[mid - 1],
                theta52[mid] - theta52[mid - 1]])
add_3d_arrowhead(ax2, tip=p_tip, direction=vec, width=0.5, height=0.6, color="#6cca31")
plot_invariant_set_sym_gen(ax2, omega[1], 0, coupling*W[1, 0], alpha[1, 0], sym_gen_vs_epsilon[epsilon_index1],
                           n, cut_value=cut_value, epsilon=1e-6, color='#6dbc3f', lw=0.5, alpha=alphaplot)  #74a655
plot_invariant_set_sym_gen(ax2, omega[1], 0, coupling*W[1, 0], alpha[1, 0], sym_gen_vs_epsilon[epsilon_index2],
                           n, cut_value=cut_value, epsilon=1e-6, color='#c3f3a6', lw=0.5, alpha=alphaplot-0.15)
print(sym_gen_vs_epsilon[epsilon_index1], sym_gen_vs_epsilon[epsilon_index2])
print(epsilon_array[epsilon_index2])
# ax2.set_title(r'Invariant sets of $S_2[c_{2345}]$')
ax2.grid(False)
ax2.view_init(elev=45, azim=40)   # 31, 43
ax2_labels = [r'$\theta_2$', r'$\theta_4$', r'$\theta_5$']
draw_axes_at_origin_with_pi_ticks(ax2, ax2_labels, axis_length=2*np.pi, tick_spacing=np.pi, offset=offset2, fontsize=10)

plt.tight_layout()
save_path = path / (file_name.removesuffix("parameters_dictionary.json") + "surfaces" + ".pdf")
plt.savefig(save_path, bbox_inches='tight')
plt.show()
