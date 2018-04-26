from __future__ import division
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


def zoom_sampling_plot():
    regular_timeseries = import_timeseries_from_file('regular_sampling.txt')
    random_timeseries = import_timeseries_from_file('random_sampling.txt')
    irregular_timeseries = import_timeseries_from_file('irregular_sampling.txt')

    regular_timeseries = np.fromiter((t for t in regular_timeseries if t < PERIOD_1 / 12.0), dtype=np.float64)
    random_timeseries = random_timeseries[:len(regular_timeseries)]
    irregular_timeseries = irregular_timeseries[:len(regular_timeseries)]

    regular_sine = sine(regular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    random_sine = sine(random_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    irregular_sine = sine(irregular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)

    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_tick_params(labelsize=SMALL_SIZE)

    ax.set_ylabel('Time series value', fontsize=MEDIUM_SIZE)
    ax.set_xlabel('Time (days)', fontsize=MEDIUM_SIZE)

    OFFSET = 0.5

    ax.scatter(regular_timeseries, regular_sine + OFFSET, s=0.5, label='Regular Sampling', c='k')
    ax.scatter(random_timeseries, random_sine, s=0.5, label='Random Sampling', c='r')
    ax.scatter(irregular_timeseries, irregular_sine - OFFSET, s=0.5, label='Irregular Sampling', c='b')

    data_lgnd = ax.legend(scatterpoints=1, loc='best', fontsize=SMALL_SIZE)

    for handle in data_lgnd.legendHandles:
        handle.set_sizes([6.0])

    plt.show()

    fig.savefig('Zoomed Data Plot (single sine wave {} period).pdf'.format(PERIOD_1))


def same_plot_with_offset():
    regular_timeseries = import_timeseries_from_file('regular_sampling.txt')
    random_timeseries = import_timeseries_from_file('random_sampling.txt')
    irregular_timeseries = import_timeseries_from_file('irregular_sampling.txt')

    # regular_timeseries = regular_timeseries[0::40]
    # random_timeseries = random_timeseries[0::40]
    # irregular_timeseries = irregular_timeseries[0::40]

    regular_sine = sine(regular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    random_sine = sine(random_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)
    irregular_sine = sine(irregular_timeseries, period=PERIOD_1, amplitude=AMPLITUDE_1, phase=PHASE_1)

    regular_sine2 = sine(regular_timeseries, period=PERIOD_2, amplitude=AMPLITUDE_2, phase=PHASE_2)
    random_sine2 = sine(random_timeseries, period=PERIOD_2, amplitude=AMPLITUDE_2, phase=PHASE_2)
    irregular_sine2 = sine(irregular_timeseries, period=PERIOD_2, amplitude=AMPLITUDE_2, phase=PHASE_2)

    regular_sin_sum = regular_sine + regular_sine2
    random_sin_sum = random_sine + random_sine2
    irregular_sin_sum = irregular_sine + irregular_sine2

    regular_sin_correlation, _ = find_correlation_from_lists(regular_timeseries, regular_sine)
    random_sin_correlation, _ = find_correlation_from_lists(random_timeseries, random_sine)
    irregular_sin_correlation, _ = find_correlation_from_lists(irregular_timeseries, irregular_sine)

    regular_sin_sum_correlation, _ = find_correlation_from_lists(regular_timeseries, regular_sin_sum)
    random_sin_sum_correlation, _ = find_correlation_from_lists(random_timeseries, random_sin_sum)
    irregular_sin_sum_correlation, _ = find_correlation_from_lists(irregular_timeseries, irregular_sin_sum)

    ####### SINGLE SIN FIGURE PLOT ######

    single_sine_fig = plt.figure(figsize=(11.7, 8.3))

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # single_sine_fig.suptitle('A single sine wave of period {} days with different samplings and '
    #                          'its autocorrelation function'.format(PERIOD_1))

    data_ax = single_sine_fig.add_subplot(2, 1, 1)
    col_ax = single_sine_fig.add_subplot(2, 1, 2)
    data_ax.xaxis.set_tick_params(labelsize=SMALL_SIZE)
    col_ax.xaxis.set_tick_params(labelsize=SMALL_SIZE)

    data_ax.set_ylabel('Time series value', fontsize=MEDIUM_SIZE)
    data_ax.set_xlabel('Time (days)', fontsize=MEDIUM_SIZE)

    col_ax.set_ylabel('G-ACF', fontsize=MEDIUM_SIZE)
    col_ax.set_xlabel('Generalised Lag (days)', fontsize=MEDIUM_SIZE)

    # col_ax.yaxis.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    # col_ax.yaxis.set_ticklabels(['-1', '0', '-1 / 1', '0', '-1 / 1', '0', '1'])

    OFFSET = 2.0

    data_ax.scatter(regular_timeseries, regular_sine + OFFSET, s=0.1, label='Regular Sampling', c='k')
    data_ax.scatter(random_timeseries, random_sine, s=0.1, label='Random Sampling', c='r')
    data_ax.scatter(irregular_timeseries, irregular_sine - OFFSET, s=0.1, label='Irregular Sampling', c='b')

    col_ax.scatter(regular_sin_correlation['lag_timeseries'], np.add(regular_sin_correlation['correlations'], 0.0),
                   s=0.1, label='Regular Sampling', alpha=1, c='k')
    col_ax.scatter(random_sin_correlation['lag_timeseries'], random_sin_correlation['correlations'], s=0.1,
                   label='Random Sampling', alpha=1, c='r')
    col_ax.scatter(irregular_sin_correlation['lag_timeseries'], np.add(irregular_sin_correlation['correlations'], 0.0)
                   , s=0.1, label='Irregular Sampling', alpha=1, c='b')

    # col_ax.axvline(x=PERIOD_1, lw=0.5, c='k')

    # col_ax.axhline(OFFSET, lw=0.1, c='k')
    col_ax.axhline(0.0, lw=0.1, c='k')
    # col_ax.axhline(-OFFSET, lw=0.1, c='k')

    col_ax.set_ylim([-1.0, 1.0])

    data_lgnd = data_ax.legend(scatterpoints=1, loc=1, fontsize=SMALL_SIZE)
    col_lgnd = col_ax.legend(scatterpoints=1, loc=1, fontsize=SMALL_SIZE)

    for handle in data_lgnd.legendHandles:
        handle.set_sizes([6.0])

    for handle in col_lgnd.legendHandles:
        handle.set_sizes([6.0])

    plt.show()

    ####### TWO SIN FIGURE PLOT ######

    two_sine_fig = plt.figure(figsize=(11.7, 8.3))

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # two_sine_fig.suptitle('A signal composed of two sine waves of period {} days and {} days with '
    #                       'different samplings and its autocorrelation function'.format(PERIOD_1, PERIOD_2))

    data_ax2 = two_sine_fig.add_subplot(2, 1, 1)
    col_ax2 = two_sine_fig.add_subplot(2, 1, 2)

    data_ax2.xaxis.set_tick_params(labelsize=SMALL_SIZE)
    col_ax2.xaxis.set_tick_params(labelsize=SMALL_SIZE)
    data_ax2.set_ylabel('Time series value', fontsize=MEDIUM_SIZE)
    data_ax2.set_xlabel('Time (days)', fontsize=MEDIUM_SIZE)

    col_ax2.set_ylabel('G-ACF', fontsize=MEDIUM_SIZE)
    col_ax2.set_xlabel('Generalised Lag (days)', fontsize=MEDIUM_SIZE)

    # col_ax2.yaxis.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    # col_ax2.yaxis.set_ticklabels(['-1', '0', '-1 / 1', '0', '-1 / 1', '0', '1'])

    OFFSET2 = 2.0

    data_ax2.scatter(regular_timeseries, regular_sin_sum + OFFSET2, s=0.1, label='Regular Sampling', c='k')
    data_ax2.scatter(random_timeseries, random_sin_sum, s=0.1, label='Random Sampling', c='r')
    data_ax2.scatter(irregular_timeseries, irregular_sin_sum - OFFSET2, s=0.1, label='Irregular Sampling', c='b')

    col_ax2.scatter(regular_sin_sum_correlation['lag_timeseries'],
                    np.add(regular_sin_sum_correlation['correlations'], 0.0),
                    s=0.1, label='Regular Sampling', alpha=1, c='k')
    col_ax2.scatter(random_sin_sum_correlation['lag_timeseries'],
                    random_sin_sum_correlation['correlations'], s=0.1,
                    label='Random Sampling', alpha=1, c='r')
    col_ax2.scatter(irregular_sin_sum_correlation['lag_timeseries'],
                    np.add(irregular_sin_sum_correlation['correlations'], 0.0),
                    s=0.1, label='Irregular Sampling', alpha=1, c='b')

    # col_ax2.axvline(x=PERIOD_1, lw=0.5, c='k')
    # col_ax2.axvline(x=PERIOD_2, lw=0.5, c='k')

    # col_ax2.axhline(OFFSET2, lw=0.1, c='k')
    col_ax2.axhline(0.0, lw=0.1, c='k')
    # col_ax2.axhline(-OFFSET2, lw=0.1, c='k')

    col_ax2.set_ylim([-1.0, 1.0])

    data_lgnd2 = data_ax2.legend(scatterpoints=1, loc=1, fontsize=SMALL_SIZE)
    col_lgnd2 = col_ax2.legend(scatterpoints=1, loc=1, fontsize=SMALL_SIZE)

    for handle in data_lgnd2.legendHandles:
        handle.set_sizes([6.0])

    for handle in col_lgnd2.legendHandles:
        handle.set_sizes([6.0])

    plt.show()

    single_sine_fig.savefig('single_sine_wave_of_{}_day_period.pdf'.format(PERIOD_1))
    two_sine_fig.savefig('two_sine_waves_of_{}_day_and_{}_day_period.pdf'.format(PERIOD_1, PERIOD_2))


# periods in days
PERIOD_1 = 17.8
PERIOD_2 = 8.7

AMPLITUDE_1 = 1.0
AMPLITUDE_2 = 1.0

PHASE_1 = 0.0
PHASE_2 = 0.0

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

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


def create_sliding_timeseries_illustration(lag=0):
    import matplotlib.animation as animation
    import matplotlib as mpl

    def get_closest_idx(target, array):
        distance = np.abs(np.subtract(array, target))
        idx = distance.argmin()
        return idx

    a = 3
    fig = plt.figure(figsize=(a * 3., a))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    fig.add_axes(ax)
    static_ax = ax.twinx()

    static_ax.get_yaxis().set_visible(False)
    static_ax.get_xaxis().set_visible(True)
    static_ax.get_xaxis().tick_bottom()
    static_ax.tick_params(bottom=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    POINT_SIZE = 10
    COLOR = 'k'

    BOTTOM = -5
    TOP = 5
    LEFT = -1
    RIGHT = 14
    OFFSET = 2

    t0 = [0., 1., 2., 4., 4.5, 5., 7., 7.1, 7.8, 8., 10.]
    t0 = [t + OFFSET for t in t0]
    t1 = [t + lag for t in t0]
    max_t = max(t0)

    t_idx = [get_closest_idx(t, t0) for t in t1]
    # print t0
    # print t1
    # print t_idx
    # print np.array(t0)[t_idx]

    y0 = np.linspace(1, 1, len(t0))
    y1 = np.linspace(-1, -1, len(t1))

    plt.rc('text', usetex=True)

    ax.set_ylim(bottom=BOTTOM, top=TOP)
    static_ax.set_ylim(bottom=BOTTOM, top=TOP)
    ax.set_xlim(left=LEFT, right=RIGHT)
    static_ax.set_xlim(left=LEFT, right=RIGHT)

    # xmin, xmax = static_ax.get_xaxis().get_view_interval()
    # ymin, ymax = static_ax.get_yaxis().get_view_interval()
    # static_ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    # static_ax.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax), color='black', linewidth=2))
    # static_ax.add_artist(plt.Line2D((t0[0], t0[-1]), (BOTTOM+1.5, BOTTOM+1.5), color='black', linewidth=2))
    # static_ax.text(7, BOTTOM+0.5, "Time", ha='center', fontsize=BIGGER_SIZE)
    # for i in range(int(t0[0]), int(t0[-1]+1)):
    #     time_point = i - 2
    #     static_ax.add_artist(plt.Line2D((i, i), (BOTTOM + 1.5, BOTTOM + 1.7), color='black', linewidth=2))
    #     static_ax.text(i, BOTTOM + 2.0, time_point, fontsize=MEDIUM_SIZE)

    static_ax.scatter(t0, y0, s=POINT_SIZE, c=COLOR)
    static_ax.set_xlabel("Time")
    static_ax.annotate("Time", xytext=(LEFT + 1, BOTTOM + 1), xy=(RIGHT - 1, BOTTOM + 1),
                       arrowprops=dict(arrowstyle="->"), fontsize=BIGGER_SIZE)
    TEXT_OFFSET = 0.5
    static_ax.text(LEFT + 0.5, y0[0] - TEXT_OFFSET, r"$T_I$", fontsize=BIGGER_SIZE)
    static_ax.text(LEFT + 0.5, y1[0] - TEXT_OFFSET, r"$T_I + \hat k$", fontsize=BIGGER_SIZE)

    # ax.scatter(t1, y1, s=POINT_SIZE, c=COLOR)
    # #
    # for t, idx in zip(t1, t_idx):
    #     ax.plot([t, t0[idx]], [-1, 1], c='r')
    #
    # plt.show()

    cnorm = mpl.colors.Normalize(vmin=0., vmax=1.)
    cmap = mpl.cm.get_cmap('cubehelix')

    ax.text(0, 2, r"$\hat k =$ {:.2f}".format(0.), fontsize=BIGGER_SIZE)
    weights_sum = []
    times = []

    def animate(i):
        t1 = [t + i for t in t0 if (t + i) <= max_t]
        y1 = np.linspace(-1, -1, len(t1))
        ax.clear()
        # scat = ax.scatter(t0, y0, s=POINT_SIZE, c=COLOR)
        ax.scatter(t1, y1, s=POINT_SIZE, c=COLOR)
        # ax.scatter(t1, y1, s=POINT_SIZE, c='r')
        t_idx = [get_closest_idx(t, t0) for t in t1]
        weights = [abs(t - t0[idx]) for t, idx in zip(t1, t_idx)]
        snorm = np.interp(weights, (0, 1), (1, 0.2))

        j = 0
        for t, idx in zip(t1, t_idx):
            ax.plot([t, t0[idx]], [-1, 1], c='r')  # ,  # cmap(cnorm(weights[i])),
            # alpha=snorm[j])
            if j == 0:
                mid_x = np.average([t0[idx], t])
                mid_y = np.average([y0[0], y1[0]])
                ax.annotate(r"$\hat S$", xy=(mid_x, mid_y), xytext=(mid_x - 1, mid_y + 1.5),
                            arrowprops=dict(arrowstyle="->"), fontsize=BIGGER_SIZE)
                # ax.text(mid_x, mid_y, r"$\hat S_{}$".format(i))
            j += 1
        ax.set_ylim(bottom=BOTTOM, top=TOP)
        static_ax.set_ylim(bottom=BOTTOM, top=TOP)
        ax.set_xlim(left=LEFT, right=RIGHT)
        static_ax.set_xlim(left=LEFT, right=RIGHT)

        # snorm_sum = sum(snorm)
        # weights_sum.append(snorm_sum)
        # times.append(i)

        # ax.text(0, 3, r"$\hat k =$ {:.2f}".format(i), fontsize=BIGGER_SIZE)

    ani = animation.FuncAnimation(fig, animate, np.linspace(0, 6, 100), interval=10)
    # plt.show()
    # plt.plot(times, weights_sum)
    # plt.show()

    ani.save('timeseries_lag_animation.gif', writer='imagemagick', dpi=300)
    # ani.save('timeseries_lag_animation.avi', writer='ffmpeg', fps=60)

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=60, codec='ffv1')

    # with writer.saving()


def standard_autocorrelation_plot():
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        result = result / max(result)
        return result[int(result.size / 2):]

    timeseries = import_timeseries_from_file('regular_sampling.txt')
    regular_sine = sine(timeseries, PERIOD_1)

    correlations = autocorr(regular_sine)

    fig = plt.figure(figsize=(11.7, 8.3))

    data_ax = fig.add_subplot(2, 1, 1)
    col_ax = fig.add_subplot(2, 1, 2)
    data_ax.xaxis.set_tick_params(labelsize=SMALL_SIZE)
    col_ax.xaxis.set_tick_params(labelsize=SMALL_SIZE)

    data_ax.set_ylabel('Time series value', fontsize=MEDIUM_SIZE)
    data_ax.set_xlabel('Time (days)', fontsize=MEDIUM_SIZE)

    col_ax.set_ylabel('ACF', fontsize=MEDIUM_SIZE)
    col_ax.set_xlabel('Lag (days)', fontsize=MEDIUM_SIZE)

    data_ax.scatter(timeseries, regular_sine, s=0.5, label='Regular Sampling', c='k')
    col_ax.scatter(timeseries, correlations, s=0.5, label='Regular Sampling', c='k')

    fig.savefig('single_sine_wave_ACF_period_{}_day.pdf'.format(PERIOD_1))

    plt.show()
    
    
    
    


if __name__ == '__main__':
    # for i in np.linspace(0, 9, 10):
    create_sliding_timeseries_illustration(0)
    # standard_autocorrelation_plot()
    # zoom_sampling_plot()
    # same_plot_with_offset()
    # star_spot_gp_plot(from_pickle=False)

    # separate lines more to show scale better
    # instead of legend just label lines
