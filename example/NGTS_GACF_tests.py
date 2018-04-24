from GACF import find_correlation_from_lists, find_correlation_from_file, GACF_LOG_MESSAGE
from GACF.datastructure import DataStructure
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import re
import peakutils
import os
import logging
from tqdm import tqdm
from scipy.stats import binned_statistic


# Create pseudo-random time series data.


def get_first_n_peaks(timeseries, x_values, n=3):
    timeseries = np.array(timeseries)
    x_arr = np.array(x_values)

    arg_max = np.argmax(x_arr)

    peak_idx = [arg_max]
    peak_times = [timeseries[arg_max]]
    peak_values = [x_arr[arg_max]]

    for i in xrange(1, n):
        prev_value = peak_values[i - 1]
        prev_arg_max = peak_idx[i - 1]
        arg_max = np.argmin(prev_value - timeseries[prev_arg_max:])
    # TODO - continue


def get_first_n_peaks_derivative(regular_timeseries, x_values, n=3):
    deriv = np.gradient(x_values, regular_timeseries)
    second_deriv = np.gradient(deriv, regular_timeseries)
    maxima = second_deriv[np.where(second_deriv < 0.)]

    return deriv, second_deriv, maxima


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


def create_sampled_data():
    n_values = 1000
    # average_timestep = 5.0 / 24.0 / 60.0  # 5 minute resolution
    # max_time = n_values * average_timestep * 3
    max_time = 30.
    average_timestep = max_time / float(n_values)
    time_series = []

    time_period = 2.759  # (days)
    amplitude = 0.01
    time_period2 = 3.468
    amplitude2 = 0.01
    n_periods = int(np.floor(max_time / time_period))

    for i in xrange(n_values):
        # if i*average_timestep > 10:
        #     break
        time_series.append(i * average_timestep + ((np.random.ranf() * average_timestep) + average_timestep / 2))

    time_series = add_data_gaps(time_series, 2. / 3., 1)  # mimic NGTS sampling

    phase_1 = np.random.ranf() * 2 * np.pi
    phase_2 = np.random.ranf() * 2 * np.pi

    x_values = [amplitude * np.sin(t * (2 * np.pi / time_period) + phase_1)
                + amplitude2 * np.sin(t * (2 * np.pi / time_period2) + phase_2)
                for t in time_series]

    return time_series, x_values


def get_data_from_file(filename):
    ds = DataStructure(filename)
    if ds.N_datasets == 1:
        data = ds.data()[0]
        errors = ds.errors()[0]
    else:
        data = ds.data()
        errors = ds.errors()
    return ds.timeseries(), data, errors


def create_phase(time_series, period, epoch):
    return np.mod(np.array(time_series) - epoch, period) / period


def rescale_phase(phase, offset=0.2):
    return [p - 1 if p > 1 - offset else p for p in phase]


def append_to_phase(phase, data, amt=0.05):
    indexes_before = [i for i, p in enumerate(phase) if p > 1 - amt]
    indexes_after = [i for i, p in enumerate(phase) if p < amt]

    phase_before = [phase[i] - 1 for i in indexes_before]
    data_before = [data[i] for i in indexes_before]

    phase_after = [phase[i] + 1 for i in indexes_after]
    data_after = [data[i] for i in indexes_after]

    return np.concatenate((phase_before, phase, phase_after)), \
           np.concatenate((data_before, data, data_after))


def remove_transit(time_series, data, period, epoch, plot=False):
    # phase fold
    phase = create_phase(time_series, period, epoch)
    epoch_phase = np.mod(0, period) / period
    rescaled_phase = rescale_phase(phase)

    phase_width = 0.04

    transit_phase = [p for p in rescaled_phase if -phase_width < p < phase_width]
    transit_indexes = [p[0] for p in enumerate(rescaled_phase) if -phase_width < p[1] < phase_width]
    transit_data = [data[i] for i in transit_indexes]
    transit_times = [time_series[i] for i in transit_indexes]

    if plot:
        plt.scatter(rescaled_phase, data, s=0.1)
        plt.scatter(transit_phase, transit_data, s=0.1)
        plt.axvline(x=epoch_phase)
        plt.show()

    non_transit_phase = [rescaled_phase[i] for i in range(0, len(rescaled_phase)) if i not in transit_indexes]
    non_transit_data = [data[i] for i in range(0, len(data)) if i not in transit_indexes]
    non_transit_times = [time_series[i] for i in range(0, len(time_series)) if i not in transit_indexes]
    if plot:
        plt.scatter(non_transit_phase, non_transit_data, s=0.1)
        plt.show()

    return non_transit_times, non_transit_data


def normalise_data_by_polynomial(time_series, data, order=3, plot=False, filename=None):
    poly_fit = np.poly1d(np.polyfit(time_series, data, order))
    data_fit = [poly_fit(t) for t in time_series]
    norm_data = np.divide(np.array(data), np.array(data_fit))

    if plot:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(time_series, data, s=0.1, label='Data', alpha=0.5)
        ax.plot(time_series, data_fit, c='k', label=str(poly_fit))
        ax.scatter(time_series, norm_data, s=0.1, label='Normalised Data')

        ax.set_title('Data with polynomial fit')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalised Intensity')

        ax.legend(fontsize=8)

        fig.savefig(filename + '/data_normalisation.pdf')

        plt.close(fig)

    return norm_data


def normalise_data_by_spline(time_series, data, plot=False, filename=None, **kwargs):
    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(time_series, data, **kwargs)
    data_fit = spl(time_series)
    norm_data = np.divide(np.array(data), np.array(data_fit))

    if plot:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(time_series, data, s=0.1, label='Data', alpha=0.5)
        ax.plot(time_series, data_fit, c='k', label='spline fit')
        ax.scatter(time_series, norm_data, s=0.1, label='Normalised Data')

        ax.set_title('Data with spline fit')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalised Intensity')

        ax.legend(fontsize=8)

        fig.savefig(filename + '/data_normalisation.pdf')

        plt.close(fig)

    return norm_data


def autocorrelation_plots(data_timeseries, data_values, correlation_timeseries, correlation_values,
                          fft_periods, fft_data, fft_indexes, fig=None, axs=[], peaks=True, max_peak=None,
                          running_max_peak=None):

    if fig is None:
        fig = plt.figure()
    # plot input data
    if not axs:
        ax1 = fig.add_subplot(311)
    else:
        ax1 = axs[0]
    # plt.subplot(311)
    if isinstance(data_values, dict):
        ax1.scatter(data_timeseries, data_values['median'], s=0.1)
        ax1.fill_between(data_timeseries, data_values['min'], data_values['max'], alpha=0.5)
    else:
        ax1.scatter(data_timeseries, data_values, s=0.1)
    ax1.set_title('Data')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Signal')

    # plot autocorrelation
    if not axs:
        ax2 = fig.add_subplot(312)
    else:
        ax2 = axs[1]
    # plt.subplot(312)
    if isinstance(correlation_values, dict):
        ax2.plot(correlation_timeseries, correlation_values['median'])
        ax2.fill_between(correlation_timeseries, correlation_values['min'], correlation_values['max'], alpha=0.5)
    else:
        ax2.plot(correlation_timeseries, correlation_values)
    ax2.set_title('Autocorrelation')
    ax2.set_xlabel('Lag Time (days)')
    ax2.set_ylabel('Correlation')
    ax2.set_xlim(xmin=0)

    # plot fourier transform
    if not axs:
        ax3 = fig.add_subplot(313)
    else:
        ax3 = axs[2]

    ax3.set_xscale('log')
    ax3.set_xlabel('Period (days)')

    if isinstance(fft_data, dict):
        ax3.plot(fft_periods, fft_data['median'], '--')
        ax3.fill_between(fft_periods, fft_data['min'], fft_data['max'], alpha=0.5)
    else:
        ax3.plot(fft_periods, fft_data, '--')
        ax3.plot(fft_periods[fft_indexes], fft_data[fft_indexes], 'r+', ms=5, mew=2,
             label='{} peaks'.format(len(fft_indexes)))
    if peaks:
        ax3.legend()
    if max_peak:
        ax3.axhline(y=max_peak, lw=0.2, ls='--')
        ax3.text(min(fft_periods), max_peak*1.2, 'threshold {}'.format(max_peak), fontsize=8)
    if running_max_peak is not None:
        ax3.fill_between(fft_periods, 0, running_max_peak, alpha=0.5)
    ax3.set_title('FFT with peak detection')
    ax3.set_xlim(xmin=0)

    fig.tight_layout()

    return fig, [ax1, ax2, ax3]


def save_autocorrelation_plots(data_timeseries, data_values, correlation_timeseries, correlation_values,
                               fft_periods, fft_data, fft_indexes, filename, max_peak=None, zoom_plots=True,
                               running_max_peak=None):

    fig, axs = autocorrelation_plots(data_timeseries, data_values, correlation_timeseries, correlation_values,
                                     fft_periods, fft_data, fft_indexes, max_peak=max_peak,
                                     running_max_peak=running_max_peak)

    fig.savefig(filename + '/autocorrelation.pdf')

    # zoomed autocol plot
    if zoom_plots:

        autocol_fig, autocol_ax = plt.subplots()
        ft_fig, ft_ax = plt.subplots()

        autocol_ax.plot(correlation_timeseries, correlation_values)
        autocol_ax.set_title('Autocorrelation function')
        autocol_ax.set_ylabel('Correlation')
        autocol_ax.set_xlabel('Lag Time (days)')

        ft_ax.set_xscale('log')
        ft_ax.plot(fft_periods, fft_data, '--')
        ft_ax.plot(fft_periods[fft_indexes], fft_data[fft_indexes], 'r+', ms=5, mew=2,
                   label='{} peaks'.format(len(fft_indexes)))
        if max_peak:
            ft_ax.axhline(y=max_peak, lw=0.2, ls='--')
            ft_ax.text(min(fft_periods), max_peak*1.2, 'threshold {}'.format(max_peak), fontsize=8,)
        if running_max_peak is not None:
            ft_ax.fill_between(fft_periods, 0, running_max_peak, alpha=0.5)
        ft_ax.set_xlabel('Period (days)')
        ft_ax.set_title('FFT with peak detection')
        ft_ax.set_xlim(xmin=0)

        autocol_fig.savefig(filename + '/autocol_zoom.pdf')
        ft_fig.savefig(filename + '/ft_zoom.pdf')
        plt.close(autocol_fig)
        plt.close(ft_fig)

    plt.close(fig)


def save_autocorrelation_plots_multi(data_timeseries, data_values, correlation_timeseries, correlation_values,
                                     fft_periods, fft_data, fft_indexes, filename, max_peak):

    fig, axs = None, []
    if isinstance(data_values, dict):
        fig, ax = autocorrelation_plots(data_timeseries, data_values, correlation_timeseries, correlation_values,
                                        fft_periods, fft_data, fft_indexes, peaks=False, max_peak=max_peak)
    else:
        for i, data in enumerate(data_values):
            if i == 0:
                fig, axs = autocorrelation_plots(data_timeseries, data, correlation_timeseries, correlation_values[i],
                                                 fft_periods[i], fft_data[i], fft_indexes[i], peaks=False,
                                                 max_peak=max_peak)
            else:
                _, _ = autocorrelation_plots(data_timeseries, data, correlation_timeseries, correlation_values[i],
                                             fft_periods[i], fft_data[i], fft_indexes[i], fig, axs, peaks=False)

    fig.savefig(filename + '/autocorrelation.pdf')

    plt.close(fig)


def bin_phase_curve(phase, data, statistic='median', bins=20):

    bin_medians, bin_edges, bin_number = binned_statistic(phase, data, statistic=statistic, bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    return bin_centers, bin_medians


def save_phase_plot(timeseries, period, epoch, data, filename, peak_percentage=None, bin=True):
    phase_app, data_app = append_to_phase(create_phase(timeseries, period, epoch), data)

    binned_phase_app, binned_data_app = bin_phase_curve(phase_app, data_app)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(phase_app, data_app, s=0.1)
    ax.scatter(binned_phase_app, binned_data_app, s=5, c='r')

    if peak_percentage is None:
        ax.set_title('Data phase folded on {0} day period'.format(period))
    else:
        ax.set_title('Data phase folded on {0} day period. {1:.1%} of max signal.'.format(period, peak_percentage))

    ax.axvline(x=0, lw=0.1, c='k', ls='--')
    ax.axvline(x=1, lw=0.1, c='k', ls='--')

    fig.savefig(filename + '/phase_fold_{}.pdf'.format(period))

    plt.close(fig)


def find_peaks_around_period(time_series, x_values, period,
                             num_peaks=3, fraction_around_period=0.5):
    confirm_periods = []
    p_diff = period * fraction_around_period
    for i in range(1, num_peaks + 1):
        centre_time = i * period
        start_time = np.argmin(abs(time_series - (centre_time - p_diff)))
        end_time = np.argmin(abs(time_series - (centre_time + p_diff)))
        if end_time == len(time_series) - 1:
            break
        if start_time != end_time:
            max_period_idx = np.argmax(x_values[start_time:end_time]) + start_time
        else:
            max_period_idx = start_time
        confirm_periods.append(time_series[max_period_idx] / i)

    peak = np.mean(confirm_periods)
    return peak, confirm_periods


def confirm_period(periods, indexes, correlation_timeseries, correlation_data,
                   num_peaks=3, fraction_around_period=0.5, plot=False, filename=None):
    peaks_x = []
    num_periods = len(indexes)
    if plot:
        if filename is None:
            raise ValueError('file name required to save plot')
        fig = plt.figure()
    for i, period in enumerate([periods[index] for index in indexes]):
        peak, confirm_periods = find_peaks_around_period(
            correlation_timeseries, correlation_data, period, num_peaks=num_peaks,
            fraction_around_period=fraction_around_period)
        if plot:
            ax = fig.add_subplot(num_periods, 1, i + 1)
            ax.plot(correlation_timeseries, correlation_data)
            ax.set_title('Found periods: {} (mean {}) from estimate {}'.format(
                confirm_periods, peak,  period),
                         fontsize=4)
            if i == num_periods - 1:
                ax.set_xlabel('Lag Time (days)')
            ax.set_ylabel('Correlation', fontsize=6)
            try:
                ax.set_xlim(xmin=0, xmax=confirm_periods[-1] * (num_peaks+1))
            except IndexError:
                pass
            for i, peak in enumerate(confirm_periods):
                ax.axvline(x=peak * (i + 1), c='r', lw=1)
                ax.axvline(x=period * (i + 1), c='g', lw=1)
        peaks_x.append(peak)

    if plot:
        try:
            fig.tight_layout()
        except ValueError:
            pass
        fig.savefig(filename + '/period_checking.pdf')
        plt.close(fig)

    return peaks_x


def fourier_transform_and_peaks(correlations, lag_timeseries, len_ft=None):

    if len_ft is None:
        len_ft = len(lag_timeseries) * 2

    complex_ft = fft.rfft(correlations, n=len_ft)
    freqs = fft.rfftfreq(len_ft, lag_timeseries[1] - lag_timeseries[0])

    periods = 1 / freqs
    ft = np.abs(complex_ft)

    # Find peaks of FFT

    indexes = peakutils.indexes(ft, thres=0.1,  # Fraction of largest peak
                                min_dist=3  # Number of data points between
                                )

    return ft, periods, indexes


def create_fourier_transform_animation(correlation_timeseries, correlation_data, filename, num_FT_steps=50):
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    num_timesteps = len(correlation_timeseries)

    FT_series_endpoints = np.linspace(0, num_timesteps, num_FT_steps).astype(int)

    fourier_transforms = []
    periods = []
    indexes = []

    for endpoint in tqdm(FT_series_endpoints[1:], desc='Calculating Fourier Transform', total=num_FT_steps):
        ft, periods_i, indexes_i = fourier_transform_and_peaks(correlation_data[:endpoint],
                                                               correlation_timeseries[:endpoint],
                                                               len_ft=num_timesteps * 2)
        fourier_transforms.append(ft)
        periods.append(periods_i)
        indexes.append(indexes_i)

    fig, (ax, ax2) = plt.subplots(2, 1)
    line, = ax.plot(periods[0], fourier_transforms[0], '--')
    peaks, = ax.plot(periods[0][indexes[0]], fourier_transforms[0][indexes[0]], 'r+', ms=5, mew=2,
                     label='{} peaks'.format(len(indexes[0])))
    autocol, = ax2.plot(correlation_timeseries, correlation_data)
    ax2.set_title('Autocorrelation')
    ax2.set_xlabel('Lag Time (days)')
    ax2.set_ylabel('Correlation')
    ax2.set_xlim(xmin=0)

    ax.set_title('Fourier Transform of Autocorrelation')
    ax.set_xscale('log')

    legend = ax.legend()

    arr1 = [None]
    arr2 = [None]

    def animate(i):
        legend = ax.legend()
        try:
            line.set_ydata(fourier_transforms[i])
            peaks.set_xdata(periods[i][indexes[i]])
            peaks.set_ydata(fourier_transforms[i][indexes[i]])
            peaks.set_label('{} peaks'.format(len(indexes[i])))
            ax.set_title('Fourier Transform of Autocorrelation ({:.0%} of data)'
                         .format(float(FT_series_endpoints[i]) / float(FT_series_endpoints[-1])))
            if arr1[0]:
                arr1[0].remove()
            if arr2[0]:
                arr2[0].remove()
            arr1[0] = ax2.axvline(x=correlation_timeseries[FT_series_endpoints[i]], c='r', lw=1)
        except IndexError:
            pass

    ax.set_xlabel('Period (Days)')
    ax.set_ylabel('Fourier Intensity')

    ax.set_ylim([0, max(fourier_transforms[-1]) * 1.2])
    fig.tight_layout()

    ani = animation.FuncAnimation(fig, animate, np.arange(1, num_FT_steps), blit=False)

    ani.save(filename + '/Increasing Fourier Transform.gif', dpi=80, writer='imagemagick')

    plt.close(fig)


def autocol_and_phase_folds(file, num_phase_plots=5, do_remove_transit=False, known_period=None, known_epoch=None,
                            logger=None, create_ft_animation=False, noise_threshold=None, interpolate_periods=False,
                            running_noise_threshold=None, **kwargs):
    try:
        filename = re.compile(r'[/|\\](?P<file>[^\\^/.]+)\..+$').search(file).group('file')
    except IndexError:
        filename = file

    file_dir = 'processed/' + filename

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    fh = logging.FileHandler(file_dir + '/peaks.log', 'w')
    fh.setFormatter(logging.Formatter())
    try:
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
    except TypeError:
        raise ValueError('No logger found')

    # get data from file
    time_series, data, _ = get_data_from_file(file)

    # time_series = np.array(time_series[:int(len(time_series)/2)])
    # data = np.array(data[:int(len(data)/2)])

    if do_remove_transit:
        if known_period is not None:
            time_series, data = remove_transit(time_series, data, known_period,
                                                   0. if known_epoch is None else known_epoch, plot=True)

    # data = normalise_data_by_polynomial(time_series, data, plot=True, filename=file_dir)

    # Find correlations
    max_lag = kwargs['max_lag'] if 'max_lag' in kwargs else None
    lag_resolution = kwargs['lag_resolution'] if 'lag_resolution' in kwargs else None
    correlations, corr = find_correlation_from_lists(time_series, data, max_lag=max_lag, lag_resolution=lag_resolution)

    logger.info(GACF_LOG_MESSAGE.format(no_data=len(time_series),
                                        no_lag_points=len(correlations['lag_timeseries']),
                                        lag_resolution=corr.lag_resolution))
    # Take fourier transform of data

    # # crop fourier transform (test)
    #
    # n = int(np.argmin(abs(np.array(correlations['lag_timeseries']) - 100.)))

    ft, periods, indexes = fourier_transform_and_peaks(correlations['correlations'], correlations['lag_timeseries'])

    logger.info('periods before pruning: {}'.format([periods[index] for index in indexes]))

    indexes = [index for index in indexes if periods[index] > 1 and index != 1]  # prune short periods & last point
    indexes = [index for index in indexes if periods[index] < 0.5 * (time_series[-1] - time_series[0])]
    signal_to_noise = None
    # prune periods more than half the time series
    if noise_threshold is not None:
        indexes = [index for index in indexes if ft[index] > noise_threshold]  # prune peaks below noise if given
    if running_noise_threshold is not None:
        # prune peaks below running noise if given
        indexes = [index for index in indexes if ft[index] > running_noise_threshold[index]]
        signal_to_noise = [ft[index] / running_noise_threshold[index] for index in indexes]

    indexes = [index for index, _ in sorted(zip(indexes, [ft[index] for index in indexes]),
                                            key=lambda pair: pair[1], reverse=True)]  # sort peaks by amplitude

    peak_percentages = [ft[index] / ft[indexes[0]] for index in indexes]

    logger.info('peak periods: {}'.format([periods[index] for index in indexes]))
    logger.info('peak amplitudes: {}'.format([ft[index] for index in indexes]))
    logger.info('peak percentages: {}'.format([str(p * 100) + '%' for p in peak_percentages]))
    logger.info('signal to noise {}'.format(signal_to_noise))

    # interpolate periods

    if interpolate_periods:
        interpolated_periods = confirm_period(periods, indexes,
                                              correlations['lag_timeseries'],
                                              correlations['correlations'],
                                              fraction_around_period=0.5,
                                              plot=False, filename=file_dir)

        if np.isnan(interpolated_periods).any():
            interpolated_periods = np.fromiter([periods[index] for index in indexes], dtype=float)

        logger.info('interpolated periods: {}'.format(interpolated_periods))
    else:
        interpolated_periods = np.fromiter([periods[index] for index in indexes], dtype=float)

    # save plots
    if running_noise_threshold is not None:
        save_autocorrelation_plots(time_series, data, correlations['lag_timeseries'], correlations['correlations'],
                                   periods, ft, indexes, file_dir, running_max_peak=running_noise_threshold)
    else:
        save_autocorrelation_plots(time_series, data, correlations['lag_timeseries'], correlations['correlations'],
                                   periods, ft, indexes, file_dir, max_peak=noise_threshold)

    for i, period in enumerate(interpolated_periods):#[periods[index] for index in indexes]):
        if i >= num_phase_plots:
            break  # only plot best n periods
        save_phase_plot(time_series, period, 0., data, file_dir, peak_percentages[i])  # TODO: find epoch and fold

    if create_ft_animation:
        create_fourier_transform_animation(correlations['lag_timeseries'], correlations['correlations'],
                                           filename=file_dir, num_FT_steps=50)


def find_noise_level(file, no_samples=10, **kwargs):

    try:
        filename = re.compile(r'[/|\\](?P<file>[^\\^/.]+)\..+$').search(file).group('file')
    except IndexError:
        filename = file

    file_dir = 'processed/' + filename + '/errors'

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    fh = logging.FileHandler(file_dir + '/peaks.log', 'w')
    fh.setFormatter(logging.Formatter())
    try:
        logger = kwargs['logger']
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
    except TypeError:
        raise ValueError('No logger found')

    # get data from file
    time_series, data, errors = get_data_from_file(file)

    median_data = np.median(data)
    median_rms = np.median(errors)

    # rms_over_data = median_rms / median_data

    max_peak = None

    generated_data = np.empty(shape=(no_samples, len(time_series)))

    for i in range(no_samples):
        generated_data[i] = np.array([np.random.normal(scale=rms) for rms in errors], dtype=float)

    # time_series = np.array(time_series[:int(len(time_series)/2)])
    # data = np.array(data[:int(len(data)/2)])

    # data = normalise_data_by_spline(time_series, data, plot=True, filename=file_dir, s=len(time_series)/100)

    # Find correlations
    # first_correlation, _ = find_correlation_from_lists(data, time_series)
    # lag_timeseries = first_correlation['lag_timeseries']
    #
    # generated_correlations = np.empty(shape=(no_samples, len(lag_timeseries)))

    generated_correlations, _ = find_correlation_from_lists(time_series, generated_data)
    lag_timeseries = generated_correlations['lag_timeseries']
    generated_correlations = np.array(generated_correlations['correlations'])

    # for i, data in enumerate(tqdm(generated_data, total=no_samples, desc='Calculating Noise Correlations')):
    #     generated_correlations[i] = find_correlation_from_lists(data, time_series)[0]['correlations']

    # Take fourier transform of data

    # # crop fourier transform (test)
    #
    # n = int(np.argmin(abs(np.array(correlations['lag_timeseries']) - 100.)))

    # generated_ft = generated_periods = generated_indexes \
    #     = generated_interp_periods = np.empty(shape=(no_samples, len(time_series)))

    for i, correlations in enumerate(tqdm(generated_correlations,
                                          total=no_samples, desc='Calculating Noise Fourier Transforms')):

        # correlations['correlations'] = np.multiply(np.array(correlations['correlations']), rms_over_data)

        ft, periods, indexes = fourier_transform_and_peaks(correlations, lag_timeseries)

        # logger.info('periods before pruning: {}'.format([periods[index] for index in indexes]))

        indexes = [index for index in indexes if periods[index] > 1 and index != 1]  # prune short periods & last point

        indexes = [index for index, _ in sorted(zip(indexes, [ft[index] for index in indexes]),
                                                key=lambda pair: pair[1], reverse=True)]  # sort peaks by amplitude

        peak_percentages = [ft[index] / ft[indexes[0]] for index in indexes]

        # logger.inxfo('peak periods: {}'.format([periods[index] for index in indexes]))
        # logger.info('peak amplitudes: {}'.format([ft[index] for index in indexes]))
        # logger.info('peak percentages: {}\n\n'.format([str(p * 100) + '%' for p in peak_percentages]))

        try:
            max_peak = max(max_peak, max([ft[index] for index in indexes]))
        except ValueError:
            max_peak = None

    # don't interpolate periods to save time

        if i == 0:
            generated_ft = np.empty(shape=(no_samples, len(ft)))
            generated_periods = periods
            generated_indexes = np.empty(shape=(no_samples,), dtype=object)

        generated_ft[i] = ft
        generated_indexes[i] = indexes

    gen_data_dict = {}
    gen_autocol_dict = {}
    gen_ft_dict = {}

    data_transpose = np.array([generated_data[:, i] for i, _ in enumerate(time_series)])
    gen_data_dict['max'] = [max(data) for data in data_transpose]
    gen_data_dict['min'] = [min(data) for data in data_transpose]
    gen_data_dict['median'] = [np.median(data) for data in data_transpose]

    autocol_transpose = np.array([generated_correlations[:, i] for i, _ in enumerate(lag_timeseries)])
    gen_autocol_dict['max'] = [max(autocol) for autocol in autocol_transpose]
    gen_autocol_dict['min'] = [min(autocol) for autocol in autocol_transpose]
    gen_autocol_dict['median'] = [np.median(autocol) for autocol in autocol_transpose]

    gen_ft_transpose = np.array([generated_ft[:, i] for i, _ in enumerate(generated_periods)])
    gen_ft_dict['max'] = [max(gen_ft) for gen_ft in gen_ft_transpose]
    gen_ft_dict['min'] = [min(gen_ft) for gen_ft in gen_ft_transpose]
    gen_ft_dict['median'] = [np.median(gen_ft) for gen_ft in gen_ft_transpose]

    # gen_ft_above_median = np.array([gen_ft[np.where(gen_ft > gen_ft_dict['median'][i])]
    #                                 for i, gen_ft in enumerate(gen_ft_transpose)])
    # std_dev_from_median = np.array([gen_ft - gen_ft_dict['median'][i] for i, gen_ft in enumerate(gen_ft_above_median)])
    # five_sigma = np.multiply(5.0, [np.mean(std_dev) for std_dev in std_dev_from_median])

    five_sigma = np.multiply(gen_ft_dict['max'], 5.0 / 3.0)  # multiply 3-sigma threshold by 5/3 to get 5 sigma

    # fig, ax = plt.subplots()
    #
    # ax.plot(generated_periods, gen_ft_dict['median'])
    # ax.fill_between(generated_periods, gen_ft_dict['min'], gen_ft_dict['max'], alpha=0.3)
    # ax.axhline(y=max_peak, ls='--')
    #
    # plt.show()

    # save plots
    # save_autocorrelation_plots_multi(time_series, generated_data, lag_timeseries,
    #                                  generated_correlations,
    #                                  generated_periods, generated_ft, generated_indexes, file_dir, max_peak)

    save_autocorrelation_plots_multi(time_series, gen_data_dict, lag_timeseries, gen_autocol_dict, generated_periods,
                                     gen_ft_dict, generated_indexes, file_dir, max_peak)

    logger.info('Noise Threshold: {}'.format(max_peak))
    logger.info('Running: {}'.format(five_sigma.tolist()))

    return max_peak, five_sigma


def pool_map_errors(file):
    try:
        return find_noise_level(file, logger=logging.getLogger(file), no_samples=1000)
    except Exception as e:
        print 'Caught Exception in child process for file {}'.format(file)
        traceback.print_exc()
        print()
        return None, None


def pool_map((file, noise_threshold)):
    try:
        return autocol_and_phase_folds(file, logger=logging.getLogger(file), create_ft_animation=True,
                                       running_noise_threshold=noise_threshold)
    except Exception as e:
        import traceback
        print 'Caught Exception in child process for file {}'.format(file)
        traceback.print_exc()
        print
        return None


def get_threshold_from_file(filename):

    try:
        filename = re.compile(r'[/|\\](?P<file>[^\\^/.]+)\..+$').search(filename).group('file')
    except IndexError:
        pass

    noise_threshold, noise_thresholds = None, None

    threshold = re.compile(r'^Noise Threshold: (?P<threshold>\d+\.\d+)')
    thresholds = re.compile(r'Running: \[(?P<values>.*?)\]')

    try:
        with open('processed/' + filename + '/errors/peaks.log') as f:
            l = f.readlines()
            for line in l:
                match = threshold.search(line)
                match2 = thresholds.search(line)
                try:
                    if match is not None:
                        noise_threshold = float(match.group('threshold'))
                    if match2 is not None:
                        noise_thresholds = [float(i) for i in match2.group('values').split(',')]
                except AttributeError:
                    continue
        return noise_threshold, noise_thresholds
    except IOError:
        return None, None
    except (TypeError, ValueError):
        return None, None


if __name__ == '__main__':

    import multiprocessing as mp
    import traceback
    from functools import partial

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter())
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

    file_location = '/Users/joshbriegal/GitHub/GACF/example/files/'
    files = os.listdir(file_location)

    # filter files
    files = [file_location + f for f in files if ('tbin' in f and 'NG2331' in f)]
    # logger.info('Running on files: \n{}'.format('\n'.join(files)))
    # files = ['/Users/joshbriegal/GitHub/GACF/example/files/NG0612-2518_044284_LC_tbin=10min.dat']
    pool = mp.Pool(processes=mp.cpu_count())

    # noise_thresholds = [None for i in range(len(files))]
    noise_thresholds = pool.map(pool_map_errors, files)
    # noise_thresholds = [get_threshold_from_file(f) for f in files]
    running_noise_thresholds = [n[1] for n in noise_thresholds]
    noise_thresholds = [n[0] for n in noise_thresholds]
    logger.info('Noise Thresholds: {}'.format(noise_thresholds))
    # noise_thresholds = [10.0594786394]

    pool.map(pool_map, zip(files, running_noise_thresholds))
    # except Exception as e:
    #     logger.warning('Failed for {} \n Exception: {}'.format(file, e))

    #
    # for file in files:
    #     # if file == '/Users/joshbriegal/GitHub/GACF/example/files/NG0522-2518_025974_LC_tbin=10min.dat':
    #     # if file == '/Users/joshbriegal/GitHub/GACF/example/files/NG1416-3056_043256_LC_tbin=10min.dat':
    #     logger.info('\n~~~~~')
    #     logger.info('Running for file {}'.format(file))
    #     logger.info('~~~~~\n')
    #     try:
    #         autocol_and_phase_folds(file, logger=logging.getLogger(file))
    #         # autocol_and_phase_folds(file, logger=logging.getLogger(file), do_remove_transit=True,
    #         #                         known_period=4.511312922, known_epoch=2457759.12161-2456658.5)
    #     except Exception as e:
    #         logger.warning('Failed for {} \n Exception: {}'.format(file, e))
