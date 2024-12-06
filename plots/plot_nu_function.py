from dynamics.symmetries import nu_function, nu_derivative
import numpy as np
from plots.config_rcparams import *

X = np.linspace(-2, 2, 1000)
nu_list = []
nup_list = []
for i in X:
    nu_list.append(nu_function(1/i))
    nup_list.append(nu_derivative(i))
print(nu_list)
nu_list[499] = np.nan

plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.title("(a)", fontsize=11)
plt.plot(X, nu_list, linewidth=2)
plt.vlines(0, -2, 2, linewidth=2.1, color="white")
ylab = plt.ylabel("$f(X)$")
ylab.set_rotation(0)
plt.xlabel("$X$")
plt.ylim([-2.1, 2.1])
plt.xlim([-2.1, 2.1])

plt.subplot(122)
plt.title("(b)", fontsize=11)
plt.plot(X, nup_list, linewidth=2)
ylab = plt.ylabel("$\\frac{\\mathrm{d}f(X)}{\\mathrm{d}X}$")
ylab.set_rotation(0)
plt.xlabel("$X$")
plt.ylim([-2.1, 2.1])
plt.xlim([-2.1, 2.1])

plt.show()
