from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_test():
    """
    A simple example of an animated plot
    """
    import numpy as np

    fig, ax = plt.subplots()

    x = np.arange(0, 2 * np.pi, 0.01)
    line, = ax.plot(x, np.sin(x))

    def animate(i):
        line.set_ydata(np.sin(x + i / 10.0))  # update the data
        return line,

    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                                  interval=25, blit=True)
    plt.show()


# run FT on increasing number of data points, pad out with NaNs to max length to allow plots to look good.

# 1) calc. autocol
# 2) calc. FT for N data points -> can supply same length each time?

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


if __name__ == '__main__':
    from GACF import find_correlation_from_lists
    from function_import_sine_wave_test import get_data_from_file, fourier_transform_and_peaks

    file = '/Users/joshbriegal/GitHub/GACF/example/files/0409-1941_004352_LC_tbin=10min.dat'
    filename =

    time_series, data, _ = get_data_from_file(file)

    correlations = find_correlation_from_lists(data, time_series)

    num_timesteps = len(correlations['lag_timeseries'])

    num_FT_steps = 50

    FT_series_endpoints = np.linspace(0, num_timesteps, num_FT_steps).astype(int)

    fourier_transforms = []
    periods = []
    indexes = []

    for endpoint in tqdm(FT_series_endpoints[1:], desc='Calculating Fourier Transform', total=num_FT_steps):
        ft, periods_i, indexes_i = fourier_transform_and_peaks(correlations['correlations'][:endpoint],
                                                               correlations['lag_timeseries'][:endpoint],
                                                               len_ft=num_timesteps * 2)
        fourier_transforms.append(ft)
        periods.append(periods_i)
        indexes.append(indexes_i)

    fig, (ax, ax2) = plt.subplots(2, 1)
    line, = ax.plot(periods[0], fourier_transforms[0], '--')
    peaks, = ax.plot(periods[0][indexes[0]], fourier_transforms[0][indexes[0]], 'r+', ms=5, mew=2,
                     label='{} peaks'.format(len(indexes[0])))
    autocol, = ax2.plot(correlations['lag_timeseries'], correlations['correlations'])
    ax2.set_title('Autocorrelation')
    ax2.set_xlabel('Lag Time (days)')
    ax2.set_ylabel('Correlation')
    ax2.set_xlim(xmin=0)

    ax.set_title('Fourier Transform of Autocorrelation')

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
            arr1[0] = ax2.axvline(x=correlations['lag_timeseries'][FT_series_endpoints[i]], c='r', lw=1)
        except IndexError:
            pass

    ax.set_xlabel('Period (Days)')
    ax.set_ylabel('Fourier Intensity')

    ax.set_ylim([0, max(fourier_transforms[-1])*1.2])
    fig.tight_layout()

    ani = animation.FuncAnimation(fig, animate, np.arange(1, num_FT_steps), blit=False)

    ani.save('Increasing Fourier Transform.gif', dpi=80, writer='imagemagick')
