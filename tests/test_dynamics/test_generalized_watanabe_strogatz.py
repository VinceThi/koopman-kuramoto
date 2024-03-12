import numpy as np
from dynamics.generalized_watanabe_strogatz import *


def test_z_dot_shape():
    omegas_z = np.array([[1],
                        [2],
                        [3]])
    z = np.array([[np.exp(3j)],
                  [np.exp(2j)],
                  [np.exp(1j)]])
    z_and_zeta = np.array([[np.exp(3j)],
                           [np.exp(2j)],
                           [np.exp(1j)],
                           [np.exp(4j)],
                           [np.exp(5j)]])
    adj_submatrix = np.array([[0, 3, 0, 0, 2],
                              [1, 0, 1, 1, 0],
                              [6, 0, 0, 1, 0]])
    result = z_dot(z, omegas_z, adj_submatrix, z_and_zeta)

    print("test_z_dot_shape result :", result)
    assert result.shape == (3, 1)
