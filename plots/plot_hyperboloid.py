from plots.config_rcparams import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

"""
# Define the range for X and Y
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)

# Calculate R from the equation \pm \sqrt(X^2 + Y^2 - 1)
Rup = np.sqrt(X**2 + Y**2 - 1)
Rdown = -np.sqrt(X**2 + Y**2 - 1)

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot both positive and negative branches
ax.plot_surface(X, Y, Rup, alpha=0.6, cmap='viridis')
# ax.plot_surface(X, Y, Rdown, alpha=0.6, cmap='viridis')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('R')
# ax.set_title('Surface plot of $X^2 + Y^2 - R^2 = 1$')

plt.show()
"""

# From hyperboloid to disk
# """
# Generate points on the hyperboloid with one sheet: x^2 + y^2 - z^2 = 1
u = np.linspace(0, 2*np.pi, 100)  # Angle parameter
v = np.linspace(0, 2, 100)  # Vertical parameter

U, V = np.meshgrid(u, v)
X = np.cosh(V) * np.cos(U)
Y = np.cosh(V) * np.sin(U)
Z = np.sinh(V)

# Apply stereographic projection to project points to the unit disk
proj_X = X / (1 + Z)
proj_Y = Y / (1 + Z)

# Plotting the hyperboloid and projection
fig = plt.figure(figsize=(5, 5))

# 3D plot of the hyperboloid
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z, color='lightblue', alpha=0.7, rstride=5, cstride=5, edgecolor='k')
ax1.set_title('Hyperboloid (One Sheet)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# # 2D plot of the stereographic projection onto the unit disk
# ax2 = fig.add_subplot(122)
# ax2.scatter(proj_X, proj_Y, s=1, alpha=0.5, color='blue')
# circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1)
# ax2.add_artist(circle)
# ax2.set_xlim([-1.2, 1.2])
# ax2.set_ylim([-1.2, 1.2])
# ax2.set_aspect('equal')
# ax2.set_title('Stereographic Projection to Unit Disk')
# ax2.set_xlabel('u')
# ax2.set_ylabel('v')

plt.show()
"""

# From disk to hyperboloid
# Generate points on the unit disk
u = np.linspace(-1, 1, 100)  # Angle parameter
v = np.linspace(-1, 1, 100)  # Vertical parameter

U, V = np.meshgrid(u, v)
X = np.cosh(V) * np.cos(U)
Y = np.cosh(V) * np.sin(U)
Z = np.sinh(V)

# Apply inverse stereographic projection to project points to the unit disk
proj_X = X / (1 + Z)
proj_Y = Y / (1 + Z)

# Plotting the hyperboloid and projection
fig = plt.figure(figsize=(14, 7))

# 3D plot of the hyperboloid
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, color='lightblue', alpha=0.7, rstride=5, cstride=5, edgecolor='k')
ax1.set_title('Hyperboloid (One Sheet)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 2D plot of the stereographic projection onto the unit disk
ax2 = fig.add_subplot(122)
ax2.scatter(proj_X, proj_Y, s=1, alpha=0.5, color='blue')
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1)
ax2.add_artist(circle)
ax2.set_xlim([-1.2, 1.2])
ax2.set_ylim([-1.2, 1.2])
ax2.set_aspect('equal')
ax2.set_title('Stereographic Projection to Unit Disk')
ax2.set_xlabel('u')
ax2.set_ylabel('v')

plt.show()
"""