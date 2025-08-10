import numpy as np

def set_pi_ticks_3d(ax, step=np.pi, pad=2, tick_length=3):
    """Set all 3D axes to show 0..2π with π-based labels and close ticks."""

    def pi_formatter(value, _):
        multiples = value / np.pi
        if np.isclose(multiples, 0):
            return "0"
        elif np.isclose(multiples, 1):
            return r"$\pi$"
        elif np.isclose(multiples, 2):
            return r"$2\pi$"
        elif multiples.is_integer():
            return fr"${int(multiples)}\pi$"
        else:
            return fr"${multiples:.1g}\pi$"

    # Apply to each axis
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(MultipleLocator(step))
        axis.set_major_formatter(FuncFormatter(pi_formatter))

    # Set axis limits
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_zlim(0, 2 * np.pi)

    # Bring ticks closer and make them shorter
    ax.tick_params(axis='x', pad=pad, length=tick_length)
    ax.tick_params(axis='y', pad=pad, length=tick_length)
    ax.tick_params(axis='z', pad=pad, length=tick_length)

    # Optional: hide background panes for a cleaner look
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
