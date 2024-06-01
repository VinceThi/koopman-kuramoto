from plots.config_rcparams import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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