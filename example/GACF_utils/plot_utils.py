import matplotlib.pyplot as plt
from lightcurve_utils import append_to_phase, bin_phase_curve, create_phase
from period_utils import fourier_transform_and_peaks


def autocorrelation_plots(data_timeseries, data_values, correlation_timeseries, correlation_values,
                          fft_periods, fft_data, fft_indexes=None, fig=None, axs=[], peaks=True, max_peak=None,
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
        ax3.text(min(fft_periods), max_peak * 1.2, 'threshold {}'.format(max_peak), fontsize=8)
    if running_max_peak is not None:
        ax3.fill_between(fft_periods, 0, running_max_peak, alpha=0.5)
    ax3.set_title('FFT with peak detection')
    ax3.set_xlim(xmin=0)

    fig.tight_layout()

    return fig, [ax1, ax2, ax3]


def save_autocorrelation_plots(data_timeseries, data_values, correlation_timeseries, correlation_values,
                               fft_periods, fft_data, fft_indexes, filename, max_peak=None, running_max_peak=None):
    fig, axs = autocorrelation_plots(data_timeseries, data_values, correlation_timeseries, correlation_values,
                                     fft_periods, fft_data, fft_indexes, max_peak=max_peak,
                                     running_max_peak=running_max_peak)
    fig.savefig(filename)
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

    ani.save(filename, dpi=80, writer='imagemagick')

    plt.close(fig)


def save_phase_plot(timeseries, period, epoch, data, filename, signal_to_noise=None):
    phase_app, data_app = append_to_phase(create_phase(timeseries, period, epoch), data)

    binned_phase_app, binned_data_app = bin_phase_curve(phase_app, data_app)

    fig, ax = plt.subplots()

    ax.scatter(phase_app, data_app, s=0.1)
    ax.scatter(binned_phase_app, binned_data_app, s=5, c='r')

    if signal_to_noise is None:
        ax.set_title('Data phase folded on {} day period'.format(period))
    else:
        ax.set_title('Data phase folded on {} day period. Signal to noise of {}.'.format(period, signal_to_noise))

    ax.axvline(x=0, lw=0.1, c='k', ls='--')
    ax.axvline(x=1, lw=0.1, c='k', ls='--')

    fig.savefig(filename)

    plt.close(fig)


def save_data_plot(timeseries, data, filename):
    fig, ax = plt.subplots()

    ax.scatter(timeseries, data, s=0.1)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Flux')

    fig.savefig(filename)

    plt.close(fig)


def save_autocorrelation_plot(correlation_timeseries, correlation_values, filename):
    fig, ax = plt.subplots()

    ax.plot(correlation_timeseries, correlation_values)
    ax.set_title('Autocorrelation function')
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Lag Time (days)')

    fig.savefig(filename)
    plt.close(fig)


def save_ft_plt(fft_periods, fft_data, fft_indexes, filename, noise_threshold=None, running_noise_threshold=None):
    fig, ax = plt.subplots()

    ax.set_xscale('log')
    ax.plot(fft_periods, fft_data, '--')
    ax.plot(fft_periods[fft_indexes], fft_data[fft_indexes], 'r+', ms=5, mew=2,
            label='{} peaks'.format(len(fft_indexes)))
    if noise_threshold:
        ax.axhline(y=noise_threshold, lw=0.2, ls='--')
        ax.text(min(fft_periods), noise_threshold * 1.2, 'threshold {}'.format(noise_threshold), fontsize=8, )
    if running_noise_threshold is not None:
        ax.fill_between(fft_periods, 0, running_noise_threshold, alpha=0.5)
    ax.set_xlabel('Period (days)')
    ax.set_title('FFT with peak detection')
    ax.set_xlim(xmin=0)

    fig.savefig(filename)
    plt.close(fig)
