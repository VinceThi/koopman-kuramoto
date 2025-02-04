import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# Define the Kuramoto model dynamics for two oscillators
def kuramoto_oscillators(t, theta, omega, K):
    dtheta = np.zeros_like(theta)
    dtheta[0] = omega[0] + K * np.sin(theta[1] - theta[0])
    dtheta[1] = omega[1] + K * np.sin(theta[0] - theta[1])
    return dtheta

# Parameters for the model
omega = [0.5, 1]  # Natural frequencies of the oscillators
K = 1  # Coupling strength

# Create a grid for the 2-torus (theta1, theta2)
N = 20
theta1 = np.linspace(0, 2 * np.pi, N)
theta2 = np.linspace(0, 2 * np.pi, N)
Theta1, Theta2 = np.meshgrid(theta1, theta2)

# Compute the vector field
dTheta1 = np.zeros_like(Theta1)
dTheta2 = np.zeros_like(Theta2)
for i in range(N):
    for j in range(N):
        theta = [Theta1[i, j], Theta2[i, j]]
        dtheta = kuramoto_oscillators(0, theta, omega, K)
        dTheta1[i, j] = dtheta[0]
        dTheta2[i, j] = dtheta[1]

# Convert to 3D coordinates on a torus
def torus_coordinates(theta1, theta2, R=3, r=1):
    X = (R + r * np.cos(theta2)) * np.cos(theta1)
    Y = (R + r * np.cos(theta2)) * np.sin(theta1)
    Z = r * np.sin(theta2)
    return X, Y, Z

X, Y, Z = torus_coordinates(Theta1, Theta2)
U, V, W = torus_coordinates(Theta1 + dTheta1, Theta2 + dTheta2)
U -= X
V -= Y
W -= Z

# Normalize vectors for consistent arrow lengths
magnitude = np.sqrt(U**2 + V**2 + W**2)
U /= magnitude
V /= magnitude
W /= magnitude

# Simulate a trajectory of the Kuramoto model using solve_ivp
def trajectory_model(t, theta):
    return kuramoto_oscillators(t, theta, omega, K)

initial_theta = [0.5, 1.0]  # Initial angles
solution = solve_ivp(trajectory_model, [0, 3], initial_theta, t_eval=np.linspace(0, 3, 1000))
trajectory = solution.y.T

# Convert trajectory to toroidal coordinates
traj_X, traj_Y, traj_Z = torus_coordinates(trajectory[:, 0], trajectory[:, 1])

# Plot the torus surface for reference
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, 2 * np.pi, 100)
Phi, Theta = np.meshgrid(phi, theta)
X_torus, Y_torus, Z_torus = torus_coordinates(Phi, Theta)
ax.plot_surface(X_torus, Y_torus, Z_torus, color='lightgray', edgecolor='none', alpha=0.6)


# Plot the vector field on a torus
ax.quiver(X, Y, Z, U, V, W, length=0.15, color='#2b2b2b')

# Plot the trajectory on the torus
ax.plot(traj_X, traj_Y, traj_Z, color='#404040', linewidth=2)

# Add a red arrowhead at the end of the trajectory line with matching linewidth
# arrow_dx = traj_X[-1] - traj_X[-2]
# arrow_dy = traj_Y[-1] - traj_Y[-2]
# arrow_dz = traj_Z[-1] - traj_Z[-2]
# arrow_magnitude = np.sqrt(arrow_dx**2 + arrow_dy**2 + arrow_dz**2)
# arrow_dx /= arrow_magnitude
# arrow_dy /= arrow_magnitude
# arrow_dz /= arrow_magnitude
# ax.quiver(
#     traj_X[-4], traj_Y[-4], traj_Z[-4],
#     arrow_dx, arrow_dy, arrow_dz,
#     color='black', arrow_length_ratio=0.1, linewidth=2
# )

ax.set_axis_off()
plt.show()
