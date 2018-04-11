import numpy as np
import matplotlib.pyplot as plt
from GACF import find_correlation_from_lists
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes


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


def sine(time, period, amplitude=1.0, phase=0.0):
    return amplitude * np.sin((time * (2 * np.pi / period)) + phase)


def separate_plots():

    regular_timeseries = import_timeseries_from_file('regular_sampling.txt')
    random_timeseries = import_timeseries_from_file('random_sampling.txt')
    irregular_timeseries = import_timeseries_from_file('irregular_sampling.txt')

    regular_sine = sine(regular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    random_sine = sine(random_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    irregular_sine = sine(irregular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)

    regular_sin_correlation, _ = find_correlation_from_lists(regular_sine, regular_timeseries)
    random_sin_correlation, _ = find_correlation_from_lists(random_sine, random_timeseries)
    irregular_sin_correlation, _ = find_correlation_from_lists(irregular_sine, irregular_timeseries)

    # plotting

    single_sine_fig = plt.figure(figsize=(11.7, 8.3))
    single_sine_fig2 = plt.figure(figsize=(11.7, 8.3))
    single_sine_fig3 = plt.figure(figsize=(11.7, 8.3))

    regular_sin_ax = single_sine_fig.add_subplot(2, 1, 1)
    random_sin_ax = single_sine_fig2.add_subplot(2, 1, 1)
    irregular_sin_ax = single_sine_fig3.add_subplot(2, 1, 1)

    regular_col_ax = single_sine_fig.add_subplot(2, 1, 2)
    random_col_ax = single_sine_fig2.add_subplot(2, 1, 2)
    irregular_col_ax = single_sine_fig3.add_subplot(2, 1, 2)

    # 6 on 1 plot
    # regular_sin_ax = single_sine_fig.add_subplot(2, 3, 1)
    # random_sin_ax = single_sine_fig.add_subplot(2, 3, 2)
    # irregular_sin_ax = single_sine_fig.add_subplot(2, 3, 3)
    #
    # regular_col_ax = single_sine_fig.add_subplot(2, 3, 4)
    # random_col_ax = single_sine_fig.add_subplot(2, 3, 5)
    # irregular_col_ax = single_sine_fig.add_subplot(2, 3, 6)

    regular_sin_ax.set_title('Regular Sampling')
    random_sin_ax.set_title('Random Sampling')
    irregular_sin_ax.set_title('Irregular Sampling')

    regular_sin_ax.set_xlabel('Time (days)')
    random_sin_ax.set_xlabel('Time (days)')
    irregular_sin_ax.set_xlabel('Time (days)')

    regular_col_ax.set_xlabel('Lag (days)')
    random_col_ax.set_xlabel('Lag (days)')
    irregular_col_ax.set_xlabel('Lag (days)')

    regular_sin_ax.set_ylabel('Signal Power')
    regular_col_ax.set_ylabel('Autocorrelation')

    regular_sin_ax.scatter(regular_timeseries, regular_sine, s=0.1)
    random_sin_ax.scatter(random_timeseries, random_sine, s=0.1)
    irregular_sin_ax.scatter(irregular_timeseries, irregular_sine, s=0.1)

    # zoomed in plots

    regular_sin_axins = zoomed_inset_axes(regular_sin_ax, zoom=5, loc=3)
    regular_sin_axins.scatter(regular_timeseries, regular_sine, s=0.1)
    regular_sin_axins.set_xlim(0, 1)
    regular_sin_axins.set_ylim(0, 0.25)
    regular_sin_axins.xaxis.set_ticklabels([])
    regular_sin_axins.yaxis.set_ticklabels([])

    random_sin_axins = zoomed_inset_axes(random_sin_ax, zoom=5, loc=3)
    random_sin_axins.scatter(random_timeseries, random_sine, s=0.1)
    random_sin_axins.set_xlim(0, 1)
    random_sin_axins.set_ylim(0, 0.25)
    random_sin_axins.xaxis.set_ticklabels([])
    random_sin_axins.yaxis.set_ticklabels([])

    irregular_sin_axins = zoomed_inset_axes(irregular_sin_ax, zoom=10, loc=3)
    irregular_sin_axins.scatter(irregular_timeseries, irregular_sine, s=0.1)
    irregular_sin_axins.set_xlim(0, 0.5)
    irregular_sin_axins.set_ylim(0, 0.1)
    irregular_sin_axins.xaxis.set_ticklabels([])
    irregular_sin_axins.yaxis.set_ticklabels([])

    regular_col_ax.scatter(regular_sin_correlation['lag_timeseries'], regular_sin_correlation['correlations'], s=0.1)
    random_col_ax.scatter(random_sin_correlation['lag_timeseries'], random_sin_correlation['correlations'], s=0.1)
    irregular_col_ax.scatter(irregular_sin_correlation['lag_timeseries'], irregular_sin_correlation['correlations'],
                             s=0.1)

    # single_sine_fig.tight_layout()
    # single_sine_fig2.tight_layout()
    # single_sine_fig3.tight_layout()

    plt.show()


def same_plot_with_offset():

    regular_timeseries = import_timeseries_from_file('regular_sampling.txt')
    random_timeseries = import_timeseries_from_file('random_sampling.txt')
    irregular_timeseries = import_timeseries_from_file('irregular_sampling.txt')

    regular_sine = sine(regular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    random_sine = sine(random_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    irregular_sine = sine(irregular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)

    regular_sine2 = sine(regular_timeseries, period=PERIOD_2, amplitude=AMPLITUDE_2, phase=PHASE_2)
    random_sine2 = sine(random_timeseries, period=PERIOD_2, amplitude=AMPLITUDE_2, phase=PHASE_2)
    irregular_sine2 = sine(irregular_timeseries, period=PERIOD_2, amplitude=AMPLITUDE_2, phase=PHASE_2)

    regular_sin_sum = regular_sine + regular_sine2
    random_sin_sum = random_sine + random_sine2
    irregular_sin_sum = irregular_sine + irregular_sine2

    regular_sin_correlation, _ = find_correlation_from_lists(regular_sine, regular_timeseries)
    random_sin_correlation, _ = find_correlation_from_lists(random_sine, random_timeseries)
    irregular_sin_correlation, _ = find_correlation_from_lists(irregular_sine, irregular_timeseries)

    regular_sin_sum_correlation, _ = find_correlation_from_lists(regular_sin_sum, regular_timeseries)
    random_sin_sum_correlation, _ = find_correlation_from_lists(random_sin_sum, random_timeseries)
    irregular_sin_sum_correlation, _ = find_correlation_from_lists(irregular_sin_sum, irregular_timeseries)

    ####### SINGLE SIN FIGURE PLOT ######

    single_sine_fig = plt.figure(figsize=(11.7, 8.3))

    single_sine_fig.suptitle('A single sine wave of period {} days with different samplings and '
                             'its autocorrelation function'.format(PERIOD_1))

    data_ax = single_sine_fig.add_subplot(2, 1, 1)
    col_ax = single_sine_fig.add_subplot(2, 1, 2)

    data_ax.set_ylabel('Signal')
    data_ax.set_xlabel('Time (days)')

    col_ax.set_ylabel('Autocorrelation')
    col_ax.set_xlabel('Lag (days)')

    # col_ax.yaxis.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    # col_ax.yaxis.set_ticklabels(['-1', '0', '-1 / 1', '0', '-1 / 1', '0', '1'])

    OFFSET = 2.0

    data_ax.scatter(regular_timeseries, regular_sine + OFFSET, s=0.1, label='Regular Sampling')
    data_ax.scatter(random_timeseries, random_sine, s=0.1, label='Random Sampling')
    data_ax.scatter(irregular_timeseries, irregular_sine - OFFSET, s=0.1, label='Irregular Sampling')

    col_ax.scatter(regular_sin_correlation['lag_timeseries'], np.add(regular_sin_correlation['correlations'], 0.0),
                   s=0.1, label='Regular Sampling', alpha=1)
    col_ax.scatter(random_sin_correlation['lag_timeseries'], random_sin_correlation['correlations'], s=0.1,
                   label='Random Sampling', alpha=1)
    col_ax.scatter(irregular_sin_correlation['lag_timeseries'], np.add(irregular_sin_correlation['correlations'], 0.0)
                   , s=0.1, label='Irregular Sampling', alpha=1)

    # col_ax.axhline(OFFSET, lw=0.1, c='k')
    col_ax.axhline(0.0, lw=0.1, c='k')
    # col_ax.axhline(-OFFSET, lw=0.1, c='k')

    col_ax.set_ylim([-1.0, 1.0])

    data_lgnd = data_ax.legend(scatterpoints=1, loc=1)
    col_lgnd = col_ax.legend(scatterpoints=1, loc=1)

    for handle in data_lgnd.legendHandles:
        handle.set_sizes([6.0])

    for handle in col_lgnd.legendHandles:
        handle.set_sizes([6.0])

    plt.show()
    
    ####### TWO SIN FIGURE PLOT ######

    two_sine_fig = plt.figure(figsize=(11.7, 8.3))

    two_sine_fig.suptitle('A signal composed of two sine waves of period {} days and {} days with '
                          'different samplings and its autocorrelation function'.format(PERIOD_1, PERIOD_2))

    data_ax2 = two_sine_fig.add_subplot(2, 1, 1)
    col_ax2 = two_sine_fig.add_subplot(2, 1, 2)

    data_ax2.set_ylabel('Signal')
    data_ax2.set_xlabel('Time (days)')

    col_ax2.set_ylabel('Autocorrelation')
    col_ax2.set_xlabel('Lag (days)')

    # col_ax2.yaxis.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    # col_ax2.yaxis.set_ticklabels(['-1', '0', '-1 / 1', '0', '-1 / 1', '0', '1'])

    OFFSET2 = 2.0

    data_ax2.scatter(regular_timeseries, regular_sin_sum + OFFSET2, s=0.1, label='Regular Sampling')
    data_ax2.scatter(random_timeseries, random_sin_sum, s=0.1, label='Random Sampling')
    data_ax2.scatter(irregular_timeseries, irregular_sin_sum - OFFSET2, s=0.1, label='Irregular Sampling')

    col_ax2.scatter(regular_sin_sum_correlation['lag_timeseries'],
                    np.add(regular_sin_sum_correlation['correlations'], 0.0),
                    s=0.1, label='Regular Sampling', alpha=1)
    col_ax2.scatter(random_sin_sum_correlation['lag_timeseries'],
                    random_sin_sum_correlation['correlations'], s=0.1,
                    label='Random Sampling', alpha=1)
    col_ax2.scatter(irregular_sin_sum_correlation['lag_timeseries'],
                    np.add(irregular_sin_sum_correlation['correlations'], 0.0),
                    s=0.1, label='Irregular Sampling', alpha=1)

    # col_ax2.axhline(OFFSET2, lw=0.1, c='k')
    col_ax2.axhline(0.0, lw=0.1, c='k')
    # col_ax2.axhline(-OFFSET2, lw=0.1, c='k')

    col_ax2.set_ylim([-1.0, 1.0])

    data_lgnd2 = data_ax2.legend(scatterpoints=1, loc=1)
    col_lgnd2 = col_ax2.legend(scatterpoints=1, loc=1)

    for handle in data_lgnd2.legendHandles:
        handle.set_sizes([6.0])

    for handle in col_lgnd2.legendHandles:
        handle.set_sizes([6.0])

    plt.show()

    single_sine_fig.savefig('single sine wave of {} day period.pdf'.format(PERIOD_1))
    two_sine_fig.savefig('two sine waves of {} day and {} day period.pdf'.format(PERIOD_1, PERIOD_2))
    

# periods in days
PERIOD_1 = 17.8
PERIOD_2 = 8.5

AMPLITUDE_1 = 1.0
AMPLITUDE_2 = 1.0

PHASE_1 = 0.0
PHASE_2 = 0.0

from george.kernels import ExpSine2Kernel, ExpSquaredKernel
import george


class StarSpotModel(object):

    def __init__(self, theta, is_exp=False):
        if is_exp:
            theta = np.exp(theta)
        self.A = theta[0]
        self.l = theta[1]
        self.G = theta[2]
        self.sigma = theta[3]
        self.P = theta[4]
        self.gp = None

    def gp_kernel(self):
        return self.A * ExpSquaredKernel(self.l) * ExpSine2Kernel(self.G, self.P)

    def gen_gp(self):

        k = self.gp_kernel()
        gp = george.GP(k, solver=george.HODLRSolver, white_noise=self.sigma)

        return gp

    def gp_sample(self, timeseries):

        if self.gp is None:
            self.gp = self.gen_gp()

        sample = self.gp.sample(timeseries)

        return sample


def star_spot_gp_plot(same_series=True, from_pickle=False, overlay_plots=True):
    import pickle
    PICKLE_FILE = "GP_Data.pkl"

    theta = [1, PERIOD_2 * 10, 1, 0.01, PERIOD_2]  # [A, l, G, sigma, P]

    regular_timeseries = import_timeseries_from_file('regular_sampling.txt')
    random_timeseries = import_timeseries_from_file('random_sampling.txt')
    irregular_timeseries = import_timeseries_from_file('irregular_sampling.txt')

    ss = StarSpotModel(theta)

    if from_pickle:
        try:
            regular_timeseries_sample = pickle.load(file(PICKLE_FILE, 'r'))
            print 'Loaded from file {}'.format(PICKLE_FILE)
        except IOError:
            from_pickle = False
            print 'Unable to load from file {}'.format(PICKLE_FILE)

    if not from_pickle:
        regular_timeseries_sample = ss.gp_sample(regular_timeseries)
        pickle.dump(regular_timeseries_sample, file(PICKLE_FILE, 'w'))
        print 'Dumped to pickle file {}'.format(PICKLE_FILE)

    if same_series:
        from scipy.interpolate import interp1d
        f = interp1d(regular_timeseries, regular_timeseries_sample, bounds_error=False, fill_value="extrapolate")
        random_timeseries_sample = f(random_timeseries)
        irregular_timeseries_sample = f(irregular_timeseries)
    else:
        random_timeseries_sample = ss.gp_sample(random_timeseries)
        irregular_timeseries_sample = ss.gp_sample(irregular_timeseries)

    # calculate autocorrelations

    regular_timeseries_correlations, _ = find_correlation_from_lists(regular_timeseries_sample, regular_timeseries)
    random_timeseries_correlations, _ = find_correlation_from_lists(random_timeseries_sample, random_timeseries)
    irregular_timeseries_correlations, _ = find_correlation_from_lists(irregular_timeseries_sample,
                                                                       irregular_timeseries)

    if overlay_plots:
        fig, ax = plt.subplots(2, 1, figsize=(11.7, 8.3))
        data_ax = ax[0]
        cor_ax = ax[1]

        OFFSET = (max(regular_timeseries_sample) - min(regular_timeseries_sample))

        data_ax.scatter(regular_timeseries, np.add(regular_timeseries_sample, OFFSET), s=0.1,
                        label='Regular Sampling')
        data_ax.scatter(random_timeseries, random_timeseries_sample, s=0.1, label='Random Sampling')
        data_ax.scatter(irregular_timeseries, np.subtract(irregular_timeseries_sample, OFFSET),
                        s=0.1, label='Irregular Sampling')

        data_ax.set_title('Regular Sampling, Gaussian Process (theta: {})'.format(theta))

        data_ax.set_xlabel('Time (days)')
        data_ax.set_ylabel('Signal')
        cor_ax.set_xlabel('Lag (days)')
        cor_ax.set_ylabel('Autocorrelation')

        cor_ax.scatter(regular_timeseries_correlations['lag_timeseries'],
                       regular_timeseries_correlations['correlations'], s=0.1, label='Regular Sampling')
        cor_ax.scatter(random_timeseries_correlations['lag_timeseries'],
                       random_timeseries_correlations['correlations'], s=0.1, label='Random Sampling')
        cor_ax.scatter(irregular_timeseries_correlations['lag_timeseries'],
                       irregular_timeseries_correlations['correlations'], s=0.1, label='Irregular Sampling')

        cor_ax.axhline(0.0, lw=0.1, c='k')

        cor_ax.set_ylim([-1.0, 1.0])

        data_lgnd = data_ax.legend(scatterpoints=1, loc=1)
        col_lgnd = cor_ax.legend(scatterpoints=1, loc=1)

        for handle in data_lgnd.legendHandles:
            handle.set_sizes([6.0])

        for handle in col_lgnd.legendHandles:
            handle.set_sizes([6.0])

        fig.tight_layout()

        plt.show()

        fig.savefig('Gaussian Process driven signal.pdf')

    else:
        fig, ax = plt.subplots(3, 1)
        regular_ax = ax[0]
        random_ax = ax[1]
        irregular_ax = ax[2]

        regular_ax.scatter(regular_timeseries, regular_timeseries_sample, s=0.1)
        random_ax.scatter(random_timeseries, random_timeseries_sample, s=0.1)
        irregular_ax.scatter(irregular_timeseries, irregular_timeseries_sample, s=0.1)

        regular_ax.set_title('Regular Sampling, Gaussian Process (theta: {})'.format(theta))
        random_ax.set_title('Random Sampling, Gaussian Process (theta: {})'.format(theta))
        irregular_ax.set_title('Irregular Sampling, Gaussian Process (theta: {})'.format(theta))

        fig.tight_layout()

        plt.show()

        fig2, axs = plt.subplots(3, 1)
        regular_ax2 = axs[0]
        random_ax2 = axs[1]
        irregular_ax2 = axs[2]

        regular_ax2.scatter(regular_timeseries_correlations['lag_timeseries'],
                            regular_timeseries_correlations['correlations'], s=0.1)
        random_ax2.scatter(random_timeseries_correlations['lag_timeseries'],
                           random_timeseries_correlations['correlations'], s=0.1)
        irregular_ax2.scatter(irregular_timeseries_correlations['lag_timeseries'],
                              irregular_timeseries_correlations['correlations'], s=0.1)

        regular_ax2.set_title('Regular Sampling, Autocorrelation')
        random_ax2.set_title('Random Sampling, Autocorrelation')
        irregular_ax2.set_title('Irregular Sampling, Autocorrelation')

        fig2.tight_layout()

        plt.show()


if __name__ == '__main__':
    # same_plot_with_offset()
    star_spot_gp_plot(from_pickle=False)

    # separate lines more to show scale better
    # instead of legend just label lines



