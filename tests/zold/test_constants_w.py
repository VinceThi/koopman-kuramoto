import numpy as np
from dynamics.constants_of_motion import get_independent_cross_ratios_complete_graph
from tests.zold.constants_w import get_w
from plots.config_rcparams import *
import pytest


def test_get_w():
    """ Test if the function get_w satisfy the last three constraints :
    np.sum(np.sin(psi)) = 0
    np.sum(np.cos(psi)) = 0
    np.sum(psi) = 0
    The two first can be summarized by np.sum(w) = 0, we take the module of np.sum(w) since the sum should yield a
    complex number very close to 0.
    The third one is equivalent to np.real(np.prod(w)) = 1
    """
    N = 20
    plot_w = False
    init_thetas = 2*np.pi*np.random.random(N)
    init_z = np.exp(1j * init_thetas)
    cross_ratios = get_independent_cross_ratios_complete_graph(init_z)
    w = get_w(cross_ratios, N, nb_iter=500)
    if plot_w:
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(121)
        ax1.scatter(np.real(init_z), np.imag(init_z), s=5, color=deep[0])
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title("$z_1$,...,$z_N$")
        ax1 = plt.subplot(122)
        ax1.scatter(np.real(w), np.imag(w), s=5, color=deep[1])
        ax1.set_xlabel('Real Part')
        ax1.set_ylabel('Imaginary Part')
        ax1.set_title("$w_1$,...,$w_N$")
        plt.show()
    num_zero = 1e-8
    assert np.abs(np.sum(w)) < num_zero and np.abs(np.real(np.prod(w)) - 1) < num_zero


if __name__ == "__main__":
    pytest.main()
