import numpy as np
import matplotlib.pyplot as plt
from GACF import find_correlation_from_lists
import numpy.fft as fft


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


def create_sampled_data(time_period, amplitude=1, phase=None, regular_sampling=True, gappy_data=True, n_values=1000,
                        max_time=30, average_timestep=None):

    if average_timestep is not None:
        max_time = n_values * average_timestep * 3
    else:
        average_timestep = max_time / float(n_values)

    time_series = []

    for i in xrange(n_values):
        time_series.append(i * average_timestep + (
                           0. if regular_sampling else ((np.random.ranf() * average_timestep) + average_timestep / 2)))

    if phase is None:
        phase = np.random.ranf() * 2 * np.pi

    if gappy_data:
        time_series = add_data_gaps(time_series, 2. / 3., 1)

    x_values = [amplitude * np.sin(t * (2 * np.pi / time_period) + phase)
                for t in time_series]

    return time_series, x_values


def create_regular_sine(time_period, **kwargs):
    kwargs['regular_sampling'] = True
    kwargs['gappy_data'] = False
    return create_sampled_data(time_period, **kwargs)


def create_random_sine(time_period, **kwargs):
    kwargs['regular_sampling'] = False
    kwargs['gappy_data'] = False
    return create_sampled_data(time_period, **kwargs)


def create_random_gappy_sine(time_period, **kwargs):
    kwargs['regular_sampling'] = False
    kwargs['gappy_data'] = True
    return create_sampled_data(time_period, **kwargs)


def create_phase(time_series, period, epoch):
    return np.mod(np.array(time_series) - epoch, period) / period


def rescale_phase(phase, offset=0.2):
    return [p - 1 if p > 1 - offset else p for p in phase]


def find_peaks_around_period(time_series, x_values, period,
                             num_peaks=3, fraction_around_period=0.5):

    confirm_periods = []
    p_diff = period * fraction_around_period
    for i in range(1, num_peaks + 1):
        centre_time = i * period
        start_time = np.argmin(abs(time_series - (centre_time - p_diff)))
        end_time = np.argmin(abs(time_series - (centre_time + p_diff)))
        if end_time == len(time_series)-1:
            break
        if start_time != end_time:
            max_period_idx = np.argmax(x_values[start_time:end_time]) + start_time
        else:
            max_period_idx = start_time
        confirm_periods.append(time_series[max_period_idx] / i)

    peak = np.mean(confirm_periods)
    return peak, confirm_periods


def confirm_period(periods, indexes, correlation_timeseries, correlation_data, plot=False):

    peaks_x = []

    for period in [periods[index] for index in indexes]:
        peak, confirm_periods = find_peaks_around_period(
            correlation_timeseries, correlation_data, period)
        peaks_x.append(peak)

        if plot:
            plt.subplot(312)
            for i, peak in enumerate(confirm_periods):
                plt.axvline(x=peak * (i + 1), c='r', lw=1)

    return peaks_x


if __name__ == '__main__':

    import time

    time_period = 2.5
    phase = None
    epoch = 0

    # time_series, data = create_regular_sine(time_period, phase=phase)
    # time_series, data = create_random_sine(time_period, phase=phase)
    time_series, data = create_random_gappy_sine(time_period, phase=phase, n_values=600000)

    plt.subplot(311)
    plt.scatter(time_series, data, s=0.1)
    plt.title('Sampled Sine wave of period {}'.format(time_period))
    plt.xlabel('time (days)')
    plt.ylabel('output')

    start = time.time()
    correlations = find_correlation_from_lists(data, time_series,
                                               lag_resolution=1./24./12.)
    end = time.time()
    print 'Time to calculate correlations {} seconds for {} data points'\
        .format(end-start, len(time_series))
    # correlations = find_correlation_from_file(filename, max_lag=50)
    start = time.time()
    complex_ft = fft.rfft(correlations['correlations'])
    freqs = fft.rfftfreq(len(correlations['lag_timeseries']), correlations['lag_timeseries'][1] -
                         correlations['lag_timeseries'][0])
    periods = 1 / freqs
    ft = np.abs(complex_ft)

    calculated_period = periods[np.argmax(ft)]

    plt.subplot(312)
    plt.plot(correlations['lag_timeseries'], correlations['correlations'])  # , s=0.1)
    plt.title('Autocorrelation')
    plt.xlabel('Lag Time (days)')
    plt.ylabel('Correlation')
    ax = plt.gca()

    import peakutils
    from peakutils.plot import plot as pplot

    indexes = peakutils.indexes(ft, thres=0.5,  # Fraction of largest peak
                                min_dist=1  # Number of data points between
                                )
    indexes = [index for index in indexes if periods[index] > 1 and index != 1]  # prune short periods & last point
    indexes = [index for index, _ in sorted(zip(indexes, [ft[index] for index in indexes]),
                                            key=lambda pair: pair[1], reverse=True)]
    print 'peak periods', [periods[index] for index in indexes]
    print 'peak amplitudes', [ft[index] for index in indexes]

    peaks_x = confirm_period(periods, indexes, correlations['lag_timeseries'], correlations['correlations'])

    print 'interpolated periods', peaks_x

    plt.subplot(313)
    pplot(periods, ft, indexes)
    plt.title('FFT with peak detection')

    plt.tight_layout()
    plt.show()

    phase_folds = []
    num_plots = 5
    num_indexes = min(len(indexes), num_plots)

    for i, period in enumerate(peaks_x):#enumerate([periods[index] for index in indexes]):
        if i >= num_plots:
            break  # only plot best n periods
        phase_folds.append(rescale_phase(create_phase(time_series, period, epoch)))
        # phase, phaseflux, phaseflux_err, N, phi = lc.phase_fold(
        #     time_series, x_values, period, 0.2, dt=0.05)
        plt.subplot(num_indexes, 1, i + 1)
        ax = plt.gca()
        plt.scatter(phase_folds[i], data, s=0.1)
        # plt.scatter(phase+0.25, phaseflux, s=12)
        # lc.plot_phase_folded_lightcurve(ax, time_series, period, 0, x_values)
        plt.title('Data phase folded on {} day period'.format(period))
        # plt.xlim(0, 1)
        # plt.ylim(0.995, 1.005)

    plt.tight_layout()
    plt.show()

    phase_fold_correct = rescale_phase(create_phase(time_series, time_period, epoch))
    plt.scatter(phase_fold_correct, data, s=0.1)
    plt.title('Data phase folded on correct period ({} days)'.format(time_period))
    plt.show()
    end = time.time()

    print 'Time to do the rest of the stuff {} seconds'.format(end-start)

