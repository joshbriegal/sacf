import numpy as np
import matplotlib.pyplot as plt
from GACF import find_correlation_from_lists
import matplotlib.pyplot as plt


def import_timeseries_from_file(filename):
    timeseries_file = open(filename)
    timeseries_file = timeseries_file.read().split('\n')
    timeseries = []
    for t in timeseries_file:
        try:
            timeseries.append(float(t))
        except (ValueError, TypeError):
            print 'Value {} not appended'.format(t)
    return np.array(timeseries)


def gaussian_weight_function(delta_t, alpha):
    return np.exp(-(delta_t ** 2) / (2.0 * alpha ** 2))


def fractional_weight_function(delta_t, alpha):
    return 1.0 / (1.0 + (delta_t / alpha))


def sine(time, period, amplitude=1.0, phase=0.0):
    return amplitude * np.sin((time * (2 * np.pi / period)) + phase)


def get_closest_idx(target, array):
    distance = np.abs(np.subtract(array, target))
    idx = distance.argmin()
    return idx


if __name__ == '__main__':
    regular_timeseries = import_timeseries_from_file('regular_sampling.txt')
    random_timeseries = import_timeseries_from_file('random_sampling.txt')
    irregular_timeseries = import_timeseries_from_file('irregular_sampling.txt')

    lag_resolution = min(np.subtract(irregular_timeseries[1:], irregular_timeseries[:-1]))

    t0 = irregular_timeseries[:500]
    lag_resolution = min(np.subtract(t0[1:], t0[:-1]))
    max_t = max(t0)
    alpha = np.median(t0)

    k = 0
    PERIOD = 1.123

    y0 = sine(t0, PERIOD)
    correlations, corr = find_correlation_from_lists(t0, y0, lag_resolution=lag_resolution, max_lag=max_t)

    lag_timeseries = []
    sum_weightsg = []
    sum_weightsf = []
    sum_weightsn = []

    while k < max_t:
        t1 = [t + k for t in t0 if (t + k) < max_t]
        t_idx = [get_closest_idx(t, t0) for t in t1]
        weightsg = [gaussian_weight_function(abs(t - t0[idx]), alpha) for t, idx in zip(t1, t_idx)]
        weightsf = [fractional_weight_function(abs(t - t0[idx]), alpha) for t, idx in zip(t1, t_idx)]
        sum_weightsg.append(sum(weightsg))
        sum_weightsf.append(sum(weightsf))
        sum_weightsn.append(float(len(t1)))
        lag_timeseries.append(k)
        k += lag_resolution

    sum_weightsf = np.divide(sum_weightsf, sum_weightsf[0])
    sum_weightsg = np.divide(sum_weightsg, sum_weightsg[0])
    sum_weightsn = np.divide(sum_weightsn, sum_weightsn[0])

    fig, (ax, ax1, ax2, ax3) = plt.subplots(4, 1)
    ax.plot(lag_timeseries, sum_weightsg, label='Gaussian')
    ax.plot(lag_timeseries, sum_weightsf, label='Fractional')
    ax.plot(lag_timeseries, sum_weightsn, label='Autocorrelation')
    ax.legend()

    ax1.scatter(t0, y0, s=0.5)
    ax2.plot(correlations['lag_timeseries'], correlations['correlations'])
    ax3.plot(correlations['lag_timeseries'], np.divide(correlations['correlations'], sum_weightsg))

    plt.show()
