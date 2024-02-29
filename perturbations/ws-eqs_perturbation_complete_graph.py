import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dynamics.watanabe_strogatz as ws


# initial conditions and structure of the system
n_iter = 1000
n = 4
init_thetas = np.random.rand(n) * 2*np.pi - np.pi
omega = 1
omegas = np.array([omega for _ in range(n)])
a = np.ones((n, n))

# add gaussian noise
# a += np.random.normal(loc=0, scale=0.01, size=(n, n))
# remove a chosen edges
# a[0, 10] = 0

np.fill_diagonal(a, 0)
k = 1

# print(f"adjacency matrix:\n {a}")
# print(f"initial angular positions:\n {init_thetas}")
# print(f"natural frequencies:\n {omegas}")
# print(f"first row of a:\n {a[0]}")


#----------------------- Integration: ORIGINAL SYSTEM -----------------------#


# define the dynamics
def kuramoto(time, thetas):
    theta_dot = []
    for i, theta in enumerate(thetas):
        sum_sin = sum([a[i, j] * np.sin(theta2 - theta) for j, theta2 in enumerate(thetas)])
        new_theta_dot = omegas[i] + k/n * sum_sin
        theta_dot.append(new_theta_dot)
    return theta_dot

# define the order parameter
def compute_order_param(thetas):
    order_param = np.sum([np.exp(1j * theta) for theta in thetas]) / n
    r = np.abs(order_param)
    theta = np.angle(order_param)
    return r, theta


# integrate the dynamics
integrator = sp.integrate.RK45(kuramoto, 0, init_thetas, n_iter, rtol=1e-10)

time_integration = []
thetas_integration = []
time = np.linspace(0, 10, 1000)
thetas = []
time_left = time
# params_ordre = []
for i in range(n_iter):
    integrator.step()
    time_integration.append(integrator.t)
    thetas_integration.append(integrator.y)
    # params_ordre.append(compute_order_param((thetas[-1] + np.pi) % (2*np.pi) - np.pi))
    interpolant = integrator.dense_output()
    for t in filter(lambda x: x < time_integration[-1], time_left):
        thetas.append(interpolant(t))
        time_left = time_left[1:]
    if integrator.status == 'finished':
        break

thetas = np.array(thetas)
thetas_mod = (thetas + np.pi) % (2*np.pi) - np.pi
# r = [param[0] for param in params_ordre]


#-------------------------- Integration: WS EQUATIONS --------------------------#


# convert theta values to z variables
init_z = np.exp(1j * init_thetas)

# set the initial state of the WS variables according to "Mobius conversion" in Cestnik and Martens (SI)
init_Z = 1j
init_phi = 3 * np.pi / 2
init_state = np.array([ init_Z, init_phi ])
w_2 = ws.w_j(init_Z, init_phi, init_z)
print('w2', w_2)
print('w2 norm', np.abs(w_2))
w = np.exp(1j * 2 * np.arctan(init_z))# set the w constants accordingly
print('arctan', np.arctan(init_z))
print('w', w)

# define the function to pass to the scipy integrator
def ws_equations_for_integrator(time, state):
    return ws.ws_equations(state, w, omega, k, n)

# integrate the WS equations
integrator_WS = sp.integrate.RK45(ws_equations_for_integrator, 0, init_state, n_iter, rtol=1e-10)

time_WSintegration = []
state_WSintegration = []
state_WS = []
time_left = time
for i in range(n_iter):
    integrator_WS.step()
    time_WSintegration.append(integrator_WS.t)
    state_WSintegration.append(integrator_WS.y)
    interpolant_WS = integrator_WS.dense_output()
    for t in filter(lambda x: x < time_WSintegration[-1], time_left):
        state_WS.append(interpolant_WS(t))
        time_left = time_left[1:]
    if integrator_WS.status == 'finished':
        break
state_WS = np.array(state_WS)
print(state_WS.shape)
print('initial conditions: ', init_z)
print([ws.z_j(init_Z, init_phi, w_j) for w_j in w])

#----------------------------------- RESULTS -----------------------------------#

# compute the trajectories from the WS variables
z_WS = []
Z = state_WS[:, 0]
phi = state_WS[:, 1]
for i, Z_t in enumerate(Z):
    z_WS.append([ws.z_j(Z_t, phi[i], w_j) for w_j in w])
z_WS = np.array(z_WS)
print(z_WS.shape)

# compute z variables from angular positions (original system)
z = np.exp(1j * thetas)

# compute the mean absolute error
err_z = np.mean(np.abs(z - z_WS), axis=1)

err_initcond = np.mean(np.abs(z[:, 0] - z_WS[:, 0]))
print(err_initcond)

# show the evolution of the mean absolute error
plt.figure()
plt.plot(time, z[:, 0], label='z_0')
plt.plot(time, z[:, 1], label='z_1')
plt.plot(time, z_WS[:, 0], label='z_0_ws')
plt.plot(time, z_WS[:, 1], label='z_1_ws')
plt.legend()
plt.show()

