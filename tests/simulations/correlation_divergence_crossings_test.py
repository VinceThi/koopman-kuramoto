import numpy as np
import pytest
from simulations.correlation_divergence_crossings import corr_divergence_crossings


# fake data
individual_timeseries = np.array([[3, 25, 5, 6],
                                  [1, 5, 6, 8],
                                  [7, 8, 10, 2],
                                  [234, 567, 4, 8],
                                  [3, 6, 3, 2],
                                  [2, 10, 26, 15],
                                  [2, 10, 26, 16],
                                  [3, 70, 25, 0]])

cr_activity = [15, 700, 3, 0.0001, 2, 4, 1000, 1100]

bounds = [0.01, 100]


#========================= TESTS =========================#

def test_1_corr_divergence_crossings():
    time, crossings = corr_divergence_crossings(cr_activity, individual_timeseries, bounds)
    assert crossings == [1, 4, 6, 3]
    assert time == [1, 3, 6, 7]

