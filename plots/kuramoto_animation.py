import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_kuramoto_on_circle(theta, sizes, interval=100, save_path=None):
    """
    Animate phases of Kuramoto oscillators on the unit circle with groups colored differently.

    Parameters:
    - theta: array of shape (T, N), phases over time
    - sizes: 1D array of ints, group sizes (must sum to N)
    - interval: milliseconds between frames
    - save_path: optional output path (gif/mp4)
    """

    T, N = theta.shape
    assert sum(sizes) == N, "Sum of sizes must equal number of oscillators"

    # Generate group labels like [0,0,...,1,1,...,2,2,...]
    group_labels = np.repeat(np.arange(len(sizes)), sizes)

    # Get a colormap with enough distinct colors
    cmap = plt.get_cmap('tab10' if len(sizes) <= 10 else 'hsv')
    colors = cmap(group_labels)

    x_circle = np.cos(np.linspace(0, 2*np.pi, 500))
    y_circle = np.sin(np.linspace(0, 2*np.pi, 500))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_circle, y_circle, color='lightgray')
    initial_xy = np.zeros((theta.shape[1], 2))  # N x 2 zero positions
    scatter = ax.scatter(initial_xy[:, 0], initial_xy[:, 1], c=colors, s=60)

    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    def init():
        scatter.set_offsets(np.zeros((N, 2)))
        return scatter,

    def update(frame):
        x = np.cos(theta[frame])
        y = np.sin(theta[frame])
        scatter.set_offsets(np.c_[x, y])
        return scatter,

    ani = animation.FuncAnimation(
        fig, update, frames=T,
        init_func=init, blit=True, interval=interval
    )

    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow")
        else:
            ani.save(save_path, writer="ffmpeg")
    else:
        plt.show()