import sys
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
# from scipy import stats, integrate
# import matplotlib.ticker as tk
# import re

plt.rcParams.update({'font.size': 22})

sys.path.append('/appch/data/jtb34/GitHub/GACF/')
sys.path.append('/appch/data/jtb34/GitHub/GACF/NGTS')
from NGTS_Field import return_field_from_object_directory
from GACF_utils import TIME_CONVERSIONS
from copy import deepcopy


def get_aliases(freq, sampling_freq, lower_limit=None, upper_limit=None, num_aliases=3):
    aliases = []
    if lower_limit is None:
        lower_limit = freq - num_aliases * sampling_freq
    if lower_limit < 0:
        lower_limit = 0
    if upper_limit is None:
        upper_limit = freq + num_aliases * sampling_freq
    calc_plus = True
    calc_minus = True
    i = 1
    aliases.append(freq)
    while calc_plus or calc_minus:
        d = i * sampling_freq
        l = freq - d
        u = freq + d
        if l > lower_limit:
            aliases.append(l)
        else:
            calc_minus = False
        if u < upper_limit:
            aliases.append(u)
        else:
            calc_plus = False
        i += 1
    return aliases


if __name__ == '__main__':

    ROOT_DIR = '/appch/data/jtb34'
    FIELDNAME = 'NG2058-0248'
    CYCLE = 'CYCLE1807'

    field = return_field_from_object_directory(ROOT_DIR, FIELDNAME, CYCLE, silent=True)# , num_objects=100)

    data = []
    for obj in field:
        if obj.periods != []:
            data.append((obj.obj, obj.periods[0], obj.peak_signal_to_noise[0]))

    # create list of tuples (obj_id, period, threshold) FOR ALL PERIODS
    data_all = []
    for obj in field:
        if obj.periods != []:
            for i, p in enumerate(obj.periods):
                data_all.append((obj.obj, p, obj.peak_signal_to_noise[i]))

    # sort list of tuples by period
    data = sorted(data, key=lambda x: x[1])
    data_all = sorted(data_all, key=lambda x: x[1])

    data_freq = [(d[0], 1.0 / d[1], d[2]) for d in data_all]
    data_freq = sorted(data_freq, key=lambda x: x[1])

    # print data_freq

    freq_resolution = 1.0 / 1000.0
    num_bins = int((data_freq[-1][1] - data_freq[0][1]) / freq_resolution)

    binned_data = []
    dfreq = deepcopy(data_freq)
    bfreq = dfreq[0][1]
    for i in range(num_bins):

        tfreq = bfreq + freq_resolution

        bin_data = []
        for i, f in enumerate([d[1] for d in dfreq]):
            if f < tfreq:
                bin_data.append(dfreq[i])
            else:
                dfreq = dfreq[i:]
                break
        if bin_data:
            binned_data.append(bin_data)
        bfreq = tfreq

    bin_medians = []
    for bin_data in binned_data:
        median = np.median([d[1] for d in bin_data])
        bin_medians.append(median)

    freq_ordered = [x for item, x in sorted(zip(binned_data, bin_medians), key=lambda x: len(x[0]), reverse=True) if
                    x < 1.0]
    timeseries_length = np.median([(obj.timeseries_binned[-1] - obj.timeseries_binned[0])
                                   for obj in field if obj.num_observations_binned > 1])
    # print timeseries_length
    day = 1.0
    aliases1 = get_aliases(freq_ordered[0], 1. / (2. * timeseries_length), num_aliases=50)

    # fig, ax = plt.subplots(figsize=(10, 7))
    # for bin_data in binned_data:
    #     _, freqs, s2n = map(list, zip(*bin_data))
    #     ax.scatter(freqs, s2n, s=1, c='b', marker='o')
    # for alias in aliases1:
    #     ax.axvline(x=alias, lw=0.5, c='r')
    # # ax.axvline(x = freq_ordered[0], lw=0.5, c='r')
    # ax.set_xscale('log')
    # ax.set_ylabel('Ratio above 5 sigma threshold')
    # ax.set_xlabel('Frequency / (1/d)')
    # ax.set_xlim(right=1.0, left=1e-2)
    # ax.set_yscale('log')
    # plt.savefig('output.pdf')
    # plt.show()

    # recursively remove aliases until max number in a bin is N.
    N = 2
    freqs = deepcopy(freq_ordered)
    bins = np.array(deepcopy(bin_medians))
    bins_data = np.array(deepcopy(binned_data))
    recurse = True
    c_freq = freqs[0]
    s_freq = 1. / (2. * timeseries_length)
    # s_freq = 1.0

    # cull periods < 1 day:
    freq_idx, = np.where(np.array(bins) < 1.0)
    bins = list(bins[freq_idx])
    bins_data = list(bins_data[freq_idx])

    max_num_per_bin = max([len(b) for b in bins_data])
    c = 1

    while max_num_per_bin > N:

        max_num_per_bin = max([len(b) for b in bins_data])
        print max_num_per_bin, 'largest bin size'

        aliases = get_aliases(c_freq, s_freq, num_aliases=50)

        fig, ax = plt.subplots(figsize=(10, 7))
        for bin_data in bins_data:
            _, freq, s2n = map(list, zip(*bin_data))
            ax.scatter(freq, s2n, s=5, c='b', marker='o')
        for alias in aliases:
            ax.axvline(x=alias, lw=0.5, c='r')
        # ax.axvline(x = freq_ordered[0], lw=0.5, c='r')
        ax.set_xscale('log')
        ax.set_ylabel('Ratio above 5 sigma threshold')
        ax.set_xlabel('Frequency / (1/d)')
        ax.set_xlim(right=10.0, left=1e-2)
        ax.set_yscale('log')
        plt.savefig('outputs' + str(c) + '.pdf')

        bins = [(i, b) for i, b in enumerate(bins) if not np.any(np.isclose(b, aliases, freq_resolution))]
        idxs, bins = map(list, zip(*bins))
        bins_data = list(np.array(bins_data)[idxs])

        freqs = [x for item, x in sorted(zip(bins_data, bins), key=lambda x: len(x[0]), reverse=True)]
        c_freq = freqs[0]

        max_num_per_bin = max([len(b) for b in bins_data])
        c += 1

    print max_num_per_bin, 'largest bin size'
    fig, ax = plt.subplots(figsize=(10, 7))
    for bin_data in bins_data:
        _, freq, s2n = map(list, zip(*bin_data))
        ax.scatter(freq, s2n, s=5, c='b', marker='o')
    ax.set_xscale('log')
    ax.set_ylabel('Ratio above 5 sigma threshold')
    ax.set_xlabel('Frequency / (1/d)')
    ax.set_xlim(right=10.0, left=1e-2)
    ax.set_yscale('log')
    plt.savefig('output_final.pdf')

