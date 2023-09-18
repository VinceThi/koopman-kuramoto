import numpy as np


# bounds = [zero_limit, infty_limit]
def corr_divergence_crossings(cr_timeseries, indiv_timeseries, bounds):
    zero_limit, infty_limit = bounds   # extract upper and lower boundaries from argument
    time = []
    crossings = []
    for t, cr_value in enumerate(cr_timeseries):
        if (abs(cr_value) < zero_limit) or (abs(cr_value) > infty_limit):
            sorted_positions = sorted(indiv_timeseries[t, :])
            differences = [sorted_positions[i+1] - sorted_positions[i] for i, _ in enumerate(sorted_positions[:-1])]
            crossings.append(min(differences))
            time.append(t)
    return time, crossings

