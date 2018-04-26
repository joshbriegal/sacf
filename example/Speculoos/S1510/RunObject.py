import sys
import os
import multiprocessing as mp
import numpy as np
import traceback
import time
import re
import json
from astropy.io import fits
import logging
import logging.handlers
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError

logging.captureWarnings(True)

import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# on server
sys.path.append('/Users/joshbriegal/GitHub/GACF/NGTS')
from NGTS_Field import NGTSField as NF
from NGTS_Object import NGTSObject as NO
from GACF_utils import TIME_CONVERSIONS, refine_period, clear_aliases, medsig
import GACF_utils as utils
import pandas as pd
from GACF import find_correlation_from_lists_cpp

if __name__ == '__main__':

    plogger = logging.getLogger('S1510 Run')
    pch = logging.StreamHandler()
    #    pformatter = logging.Formatter('{asctime} - {name} - {levelname} - {message}')
    pformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    pch.setFormatter(pformatter)
    plogger.addHandler(pch)
    plogger.setLevel(logging.INFO)

    field = NF('S1510', 'GACF', root_file='/Users/joshbriegal/GitHub/GACF/example/Speculoos')
    field.objects = {
        0: NO(0, 'S1510', 'GACF', field_filename=field.filename),
        1: NO(1, 'S1510', 'GACF', field_filename=field.filename)
    }

    df_A = pd.read_csv('/Users/joshbriegal/GitHub/GACF/example/Speculoos/S1510/PLOT__corrected_A.csv',
                       comment='#', names=['time', 'flux', 'flux_err'], na_values='--').dropna().reset_index(drop=True)
    df_B = pd.read_csv('/Users/joshbriegal/GitHub/GACF/example/Speculoos/S1510/PLOT__corrected_B.csv',
                       comment='#', names=['time', 'flux', 'flux_err'], na_values='--').dropna().reset_index(drop=True)

    # plogger.info(df_A.head(50))

    sort_idx = np.argsort(df_A['time'].tolist())
    field[0].timeseries_binned = list(np.array(df_A['time'].tolist())[sort_idx] - df_A['time'][0])
    field[0].flux_binned = list(np.array(df_A['flux'].tolist())[sort_idx])
    field[0].num_observations_binned = len(field[0].timeseries_binned)

    sort_idx = np.argsort(df_B['time'].tolist())
    field[1].timeseries_binned = list(np.array(df_B['time'].tolist())[sort_idx] - df_B['time'][0])
    field[1].flux_binned = list(np.array(df_B['flux'].tolist())[sort_idx])
    field[1].num_observations_binned = len(field[1].timeseries_binned)


    # print np.any(np.diff(df_A['time'].tolist()) > 0)

    for obj in field:
        obj.cleaned_refined_periods = [1.0]
        # obj.bin_data()
        try:
            obj.create_logger()
        except IOError:
            plogger.exception('Unable to create log file for {}'.format(obj))
        try:
            med, sig = medsig(obj.flux_binned)
            idx_ok = abs(np.subtract(obj.flux_binned, med)) < 3 * sig
            obj.timeseries_binned = list(np.array(obj.timeseries_binned)[idx_ok])
            obj.flux_binned = list(np.array(obj.flux_binned)[idx_ok])
            obj.num_observations_binned = len(obj.flux_binned)
            obj.lag_resolution = 20.0 * TIME_CONVERSIONS['m2d']
            obj.min_lag = 0
            plogger.info('Calculating correlations for {}'.format(obj))
            obj.calculate_periods_from_autocorrelation(calculate_noise_threshold=False)
        except Exception as e:
            plogger.exception('{} not correlated'.format(obj))
            obj.is_ok = False
        else:
            plogger.info("Refining periods and removing aliases for {}".format(obj))
            try:
                fourier_ok = obj.check_fourier_ok(nlim=5)
                if fourier_ok:
                    obj.clean_periods()
                    # is_moon = obj.check_moon_detection(field.new_moon_epoch)
                    # if is_moon:
                    #     moon_ok = obj.remove_moon_and_check_ok(field.new_moon_epoch)
                    # else:
                    #     moon_ok = True
                    moon_ok = True
                if fourier_ok and moon_ok:
                    signal_ok = obj.check_signal_significance()
                else:
                    obj.cleaned_refined_periods = []  # these periods are bad
            except Exception as e:
                #            plogger.warning(traceback.format_exc())
                plogger.exception('{} not processed'.format(obj))

            try:
                obj.save_and_log_object(reduce_data=False)
            except Exception as e:
                plogger.exception('{} not saved'.format(obj))

    field.remove_bad_objects()

    plogger.info('Processed Obj IDs {}'.format(field.object_list))

    plogger.info('Plotting ...')

    # field[0].timeseries_binned = map(float, list(np.array(df_A['time'].tolist())))
    # field[0].flux_binned = map(float, list(np.array(df_A['flux'].tolist())))
    # field[0].num_observations_binned = len(field[0].timeseries_binned)
    #
    # field[1].timeseries_binned = map(float, list(np.array(df_B['time'].tolist())))
    # field[1].flux_binned = map(float, list(np.array(df_B['flux'].tolist())))
    # field[1].num_observations_binned = len(field[1].timeseries_binned)

    # height_ratios = [1, 1]
    # width_ratios = [1, 1]
    pagecount = 1
    n_per_page = 2
    gs = gridspec.GridSpec(ncols=3, nrows=n_per_page)  # , height_ratios=height_ratios, width_ratios=width_ratios)
    data_axs = []
    phase_axs = []
    table_axs = []
    page = plt.figure(figsize=(15, 10))

    for i in range(n_per_page):

        ax_data = page.add_subplot(gs[i, 0])
        ax_ft = page.add_subplot(gs[i, 1])
        ax_phase = page.add_subplot(gs[i, 2])

        try:
            obj = field[i]
        except (KeyError, IndexError) as e:
            ax_data.axis('off')
            ax_phase.axis('off')
            continue

        # p = obj.cleaned_refined_periods[0]

        p = obj.periods[0]

        ax_data.scatter(obj.timeseries_binned, obj.flux_binned, s=0.5)
        if i == 0:
            ax_data.set_title('Raw Data')
            ax_ft.set_title('Fourier Transform of G-ACF')
        ax_phase.set_title('Data Phase Folded on {} days'.format(p))
        if i == n_per_page - 1:
            ax_data.set_xlabel('Time / days')
            ax_phase.set_xlabel('Phase')
            ax_ft.set_xlabel('Period / days')
        ax_data.set_ylabel('Normalised Flux')
        ax_ft.set_ylabel('Normalised Power')

        ax_ft.plot(obj.period_axis, obj.ft / np.max(obj.ft), marker='.', ls='--', lw=0.5, markersize=0.5)
        ax_ft.axvline(x=p, lw=0.5, c='r')
        # ax_ft.set_xlim(100)
        ax_ft.set_xscale('log')

        t = obj.timeseries_binned
        f = obj.flux_binned
        try:
            phase_app, data_app = utils.append_to_phase(utils.create_phase(t, p, 0), f, 0)
            binned_phase_app, binned_data_app = utils.bin_phase_curve(phase_app, data_app)

            ndata = len(phase_app)
            idx = [i for i in range(1, ndata) if phase_app[i] < phase_app[i - 1]]
            idx.append(ndata - 1)
            nidx = len(idx)
            colours = mpl.cm.rainbow(np.r_[0:1:nidx * 1j])
            phases, fs = [], []
            j = 0
            for k, i in enumerate(idx):
                phases.append(np.array(phase_app[j:i]))
                fs.append(np.array(f[j:i]))
                ax_phase.scatter(phases[k], fs[k], marker='o', s=0.5, c=colours[k])
                j = i

            ax_phase.axvline(x=0, lw=0.5, c='k', ls='--')
            ax_phase.axvline(x=1, lw=0.5, c='k', ls='--')
        except ValueError:
            pass


    plt.show()

    plogger.info('Finished plots')
