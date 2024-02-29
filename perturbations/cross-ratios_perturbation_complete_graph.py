import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dynamics.constants_of_motion import cross_ratio_theta


# initial conditions and structure of the system
n_iter = 1000
n = 100
init_thetas = np.random.rand(n) * 2*np.pi - np.pi
omegas = np.array([1 for _ in range(n)])
a = np.ones((n, n))

# add gaussian noise
# a += np.random.normal(loc=0, scale=0.01, size=(n, n))
# remove a chosen edges
a[0, 10] = 0

np.fill_diagonal(a, 0)
k = 1

print(f"adjacency matrix:\n {a}")
print(f"initial angular positions:\n {init_thetas}")
print(f"natural frequencies:\n {omegas}")
print(f"first row of a:\n {a[0]}")


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

time = []
thetas = []
params_ordre = []
for i in range(n_iter):
    integrator.step()
    time.append(integrator.t)
    thetas.append(integrator.y)
    params_ordre.append(compute_order_param((thetas[-1] + np.pi) % (2*np.pi) - np.pi))
    if integrator.status == 'finished':
        break
thetas = np.array(thetas)
thetas_mod = (thetas + np.pi) % (2*np.pi) - np.pi
r = [param[0] for param in params_ordre]


# compute the cross-ratios
c_0123 = cross_ratio_theta(thetas[:, 0], thetas[:, 1], thetas[:, 2], thetas[:, 3])
c_1234 = cross_ratio_theta(thetas[:, 1], thetas[:, 2], thetas[:, 3], thetas[:, 4])
c_5678 = cross_ratio_theta(thetas[:, 5], thetas[:, 6], thetas[:, 7], thetas[:, 8])


# show the dynamics and the cross-ratios
plt.figure(figsize=(6, 4))
plt.plot(time, c_0123, label='$c_{0123}$')
plt.plot(time, c_1234, label='$c_{1234}$')
plt.plot(time, c_5678, label='$c_{5678}$')
plt.legend()
plt.show()







