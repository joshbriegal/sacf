from GACF import find_correlation_from_lists_cpp
import numpy as np
import matplotlib.pyplot as plt

# Create pseudo-random time series data.

if __name__ == '__main__':

    n_values = 10000
    average_timestep = 5.0 / 24.0 / 60.0  # 5 minute resolution
    time_series = []

    for i in xrange(n_values):
        time_series.append(i * average_timestep + ((np.random.ranf() * average_timestep) + average_timestep / 2))

    x_values = [np.sin(t * 2 * np.pi / 2.5) for t in time_series]

    plt.subplot(211)
    plt.scatter(time_series, x_values, s=0.05)
    plt.title('Raw Data (sin wave, randomly sampled)')
    plt.xlabel('Time (days)')
    plt.ylabel('Sin(time)')

    # use in built find_correlation_from_lists_cpp (quick method)
    lag_timeseries, correlations, corr = find_correlation_from_lists_cpp(time_series, x_values,
                                                                         lag_resolution=average_timestep)

    plt.subplot(212)
    plt.plot(lag_timeseries, correlations)
    plt.title('Autocorrelation')
    plt.xlabel('Lag Time (days)')
    plt.ylabel('Correlation')

    plt.tight_layout()
    plt.show()
