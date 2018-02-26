from GACF.datastructure import DataStructure
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def get_data_from_file(filename):
    ds = DataStructure(filename)
    return ds.timeseries(), ds.values()


if __name__ == '__main__':

    # load in NGTS file as NGTS sampling

    filename = '/Users/joshbriegal/GitHub/GACF/example/files/NG0522-2518_025520_LC_tbin=10min.dat'

    time_series, _ = get_data_from_file(filename)
    time_series = np.array(time_series)
    time_series = time_series - time_series[0]
    time_series = time_series[:4000]

    regular_timeseries = np.linspace(time_series[0], time_series[-1], len(time_series))
    delta_t = regular_timeseries[1] - regular_timeseries[0] / 2
    random_timeseries = np.fromiter([t + (np.random.ranf() * delta_t + (delta_t / 2.)) for t in regular_timeseries],
                                    dtype=type(np.random.ranf()))

    # plt.scatter(time_series, np.linspace(1, 1, len(time_series)), s=0.1)
    # plt.scatter(regular_timeseries, np.linspace(2, 2, len(time_series)), s=0.1)
    # plt.scatter(random_timeseries, np.linspace(3, 3, len(time_series)), s=0.1)

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax1.scatter(time_series, np.sin(time_series), s=0.1)
    ax1.set_title('Realistic time sampling')

    ax2 = fig.add_subplot(312)
    ax2.scatter(regular_timeseries, np.sin(regular_timeseries), s=0.1)
    ax2.set_title('Regular time sampling')

    ax3 = fig.add_subplot(313)
    ax3.scatter(random_timeseries, np.sin(random_timeseries), s=0.1)
    ax3.set_title('Random time sampling')

    fig.tight_layout()
    plt.show()

    fig2 = plt.figure()

    ax2 = fig2.add_subplot(211)

    ax2.scatter(regular_timeseries[:100], np.sin(regular_timeseries[:100]), s=0.1)
    ax2.set_title('Regular time sampling')

    ax3 = fig2.add_subplot(212)
    ax3.scatter(random_timeseries[:100], np.sin(random_timeseries[:100]), s=0.1)
    ax3.set_title('Random time sampling')

    fig2.tight_layout()
    plt.show()

    print len(time_series), len(regular_timeseries), len(random_timeseries)

    np.savetxt('regular_sampling.txt', regular_timeseries, header='time', fmt='%10.10f')
    np.savetxt('random_sampling.txt', random_timeseries, header='time', fmt='%10.10f')
    np.savetxt('realistic_sampling.txt', time_series, header='time', fmt='%10.10f')













    # file_location = '/Users/joshbriegal/GitHub/GACF/example/files/'
    # files = os.listdir(file_location)
    #
    # # filter files
    # files = [file_location + f for f in files if 'tbin' in f]
    #
    # time_series_dict = {}
    #
    # for i, file in enumerate(files):
    #
    #     time_series, _ = get_data_from_file(file)
    #     time_series = np.array(time_series)
    #     time_series = time_series - time_series[0]
    #     time_series_dict[file] = time_series
    #
    #     plt.scatter(time_series, np.linspace(i, i, len(time_series)), label=file, s=0.1)
    #
    #     print i, file
    #
    # plt.show()



