from GACF import find_correlation_from_lists_cpp
import numpy as np
import numpy.fft as fft
import peakutils  # Will require install of 'peakutils' package
import matplotlib.pyplot as plt


def fourier_transform_and_peaks(lag_timeseries, correlations, len_ft=None, n=1, power_of_2=True, pad_both_sides=True):
    if len_ft is None:
        # zero pad FFT for higher resolution output
        len_ft = len(lag_timeseries) * n

    if power_of_2:
        # add more zeros to make the length of the array a power of 2, for computational speed up
        len_ft = int(2.0 ** np.ceil(np.log2(len_ft)))

    # print 'number of samples', len_ft, '2 ** ', np.log2(len_ft)

    if pad_both_sides and n > 1:
        num_zeros = len_ft - len(lag_timeseries)
        zeros = np.zeros(num_zeros / 2)
        correlations = np.append(zeros, np.append(correlations, zeros))

    # calculate FFT of correlations
    complex_ft = fft.rfft(correlations, n=len_ft)
    freqs = fft.rfftfreq(len_ft, lag_timeseries[1] - lag_timeseries[0])

    periods = 1 / freqs
    ft = np.abs(complex_ft)

    # Find peaks of FFT
    indexes = peakutils.indexes(ft, thres=0.1,  # Fraction of largest peak
                                min_dist=3  # Number of data points between peaks, 3 ==> .:. is a peak
                                )

    return ft, periods, indexes


def find_period_from_peaks(lag_timeseries, correlations, num_peaks=3, resolution=1e-8):
    indexes = peakutils.indexes(correlations, thres=0.1, min_dist=5)

    # remove zero point
    indexes = [index for index in indexes if abs(lag_timeseries[index]) > resolution]
    powers = correlations[indexes]
    lags = lag_timeseries[indexes]

    if np.any(lags < 0.):
        # treat positive and negative separately
        idx_positive = np.where(lags > 0.)[0]
        idx_negative = np.where(lags < 0.)[0]
        lp = lags[idx_positive]
        pp = powers[idx_positive]
        ln = lags[idx_negative]
        pn = powers[idx_negative]

        sorted_lp = [p[0] for p in sorted(zip(lp, pp), key=lambda x: x[1], reverse=True)]
        sorted_ln = [-p[0] for p in sorted(zip(ln, pn), key=lambda x: x[1], reverse=True)]

        positives = [p / float((i + 1)) for i, p in enumerate(sorted_lp[:num_peaks])]
        negatives = [(p / float((i + 1))) for i, p in enumerate(sorted_ln[:num_peaks])]

        return np.average([np.average(positives), np.average(negatives)])
    else:
        sorted_lag = [p[0] for p in sorted(zip(lags, powers), key=lambda x: x[1], reverse=True)]
        return np.average([p / float((i + 1)) for i, p in enumerate(sorted_lag[:num_peaks])])


# as one function if needed
def calculate_period_from_data(timeseries, values, method='fft'):
    if method not in ['fft', 'peaks']:
        raise ValueError('Method must be fft or peaks')

    lag_timeseries, correlations, _ = find_correlation_from_lists_cpp(timeseries, values,
                                                                      lag_resolution=(timeseries[1] - timeseries[0]))

    if method == 'fft':
        ft, period_axis, indexes = fourier_transform_and_peaks(lag_timeseries, correlations, n=32)

        # Find biggest peak (probably period in most cases)
        powers = ft[indexes]
        periods = period_axis[indexes]

        sorted_periods = [p[0] for p in sorted(zip(periods, powers), key=lambda x: x[1])]
        calculated_period = periods[0]

        return calculated_period
    elif method == 'peaks':
        return find_period_from_peaks(lag_timeseries, correlations, num_peaks=3)
    else:
        raise ValueError('Method must be fft or peaks')


if __name__ == "__main__":
    # generate a sine wave of known period
    period = 20.

    timeseries = np.linspace(0, 100, 1000)
    # randomly cut 20% of data
    timeseries = np.fromiter((t for t in timeseries if np.random.rand() < 0.8), dtype=float)
    values = np.sin(timeseries * (2 * np.pi / period))

    plt.scatter(timeseries, values, s=0.5)
    plt.show()

    # calculate G-ACF
    lag_timeseries, correlations, _ = find_correlation_from_lists_cpp(timeseries, values,
                                                                      lag_resolution=(timeseries[1] - timeseries[0]))
    lag_timeseries = np.array(lag_timeseries)
    correlations = np.array(correlations)

    # calculate FFT & peaks (method 1)

    ft, period_axis, indexes = fourier_transform_and_peaks(lag_timeseries, correlations, n=32)

    # Find biggest peak (probably period in most cases)
    powers = ft[indexes]
    periods = period_axis[indexes]

    sorted_periods = [p[0] for p in sorted(zip(periods, powers), key=lambda x: x[1])]
    calculated_period = sorted_periods[0]

    print 'Inputted period: {}, outputted period {} from FFT'.format(period, calculated_period)

    # calculate period by finding peaks in G-ACF

    print 'Inputted period: {}, outputted period {} from peak'.format(period,
                                                                      find_period_from_peaks(lag_timeseries,
                                                                                             correlations,
                                                                                             num_peaks=3))
