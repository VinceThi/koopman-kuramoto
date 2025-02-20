import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource


def solid_of_revolution(radius=1, R=2, theta_res=100, phi_res=50):
    theta = np.linspace(0, 2 * np.pi, theta_res)
    phi = np.linspace(0, 2 * np.pi, phi_res)
    theta, phi = np.meshgrid(theta, phi)

    x = (R + radius * np.cos(phi)) * np.cos(theta)
    y = (R + radius * np.cos(phi)) * np.sin(theta)
    z = radius * np.sin(phi)

    return x, y, z


# Parameters for the solid of revolution
R, radius = 2, 1
x, y, z = solid_of_revolution(radius, R)

# Compute normals for shading
dx, dy, dz = np.gradient(x), np.gradient(y), np.gradient(z)
normals = np.array([dx, dy, dz])
normal_magnitude = np.linalg.norm(normals, axis=0)
normals /= normal_magnitude

# Create a light source
ls = LightSource(azdeg=45, altdeg=30)
shading = ls.shade(z, cmap=plt.cm.gray, vert_exag=0.1, blend_mode='soft')

# Plot the solid of revolution
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, facecolors=shading, rstride=1, cstride=1, linewidth=0, antialiased=True, edgecolors='none',
                shade=True)
ax.set_xlim([-R - radius, R + radius])
ax.set_ylim([-R - radius, R + radius])
ax.set_zlim([-radius, radius])
ax.axis('off')

plt.show()