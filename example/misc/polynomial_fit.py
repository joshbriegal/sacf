import numpy as np
from GACF import *
from function_import_sine_wave_test import get_data_from_file
import matplotlib.pyplot as plt

def add_data_gaps(regular_timeseries, gap_length, gap_frequency):
    if gap_length > gap_frequency:
        raise ValueError('cannot have a gap larger than the gap frequency!')
    gappy_timeseries = []
    is_gap = False
    start_time = regular_timeseries[0]
    for time in regular_timeseries:
        time_diff = time - start_time
        if time_diff < gap_length:
            is_gap = True
        else:
            if time_diff > gap_frequency:
                is_gap = True
                start_time = time
            else:
                is_gap = False
        if not is_gap:
            gappy_timeseries.append(time)
        else:
            continue
    return gappy_timeseries

time_series = np.linspace(0, 200, 5000)
print len(time_series)

# time_series = np.array(add_data_gaps(time_series, (2./3.), 1.))
# print len(time_series)

sin_1 = np.sin(time_series * (2 * np.pi / 10.311751273209941) + np.random.rand()*np.pi*2)

sin_2 = (229.16270676549303 / 199.51741057255288) * np.sin(time_series * (2 * np.pi / 21.732546733698335))
# (199.51741057255288 / 229.16270676549303) *
# sin_1 = np.sin(time_series * (2 * np.pi / 10.8))

# sin_2 = 0.5 * np.sin(time_series * (2 * np.pi / 11.2) + np.pi)

data = np.add(sin_1, sin_2)
# data = np.multiply(sin_1, sin_2)
correlations = find_correlation_from_lists(data, time_series)

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.scatter(time_series, sin_1, label='sin 1', alpha=0.2, s=0.5)
ax1.scatter(time_series, sin_2, label='sin 2', alpha=0.2, s=0.5)
ax1.scatter(time_series, data, label='sum', c='r', s=0.5)
ax1.set_title('Data')
ax1.legend()

ax2 = fig.add_subplot(212)
ax2.plot(correlations['lag_timeseries'], correlations['correlations'])
ax2.set_title('Autocorrelation')


fig.tight_layout()
plt.show()
plt.close(fig)





