import sys

# import seaborn as sns
import numpy as np
from scipy import stats, integrate
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.ticker as tk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import re
from copy import deepcopy
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import pandas as pd
import scipy.interpolate as interpolate

plt.rcParams.update({'font.size': 18})

from textwrap import wrap

import json

sys.path.append('/appch/data/jtb34/GitHub/GACF/')
sys.path.append('/home/jtb34/python27')
# print sys.path
from NGTS.NGTS_Field import return_field_from_object_directory
from NGTS.GACF_utils import TIME_CONVERSIONS
import NGTS.GACF_utils as utils

ROOT_DIR = '/appch/data/jtb34'
field_pattern = re.compile(r'^NG\d+[+|-]\d+$')
object_pattern = re.compile(r'^\d+_VERSION_(?P<test>[a-zA-Z0-9]+)$')

XMATCH_LOCATION = '/appch/data/jtb34/{}/cross_match/'
XMATCH_FILE_NAME = 'Uncut_Final_{}.fits'

TEST = 'CYCLE1807'

freq_resolution = 1.0 / 1000.0

color_cycle = ['b', 'r', 'g', 'k']


def create_data_tuples(field, freq=True):
    # create list of tuples (obj_id, period, threshold) FOR ALL PERIODS
    data_all = []
    for obj in field:
        if obj.periods != []:
            for i, p in enumerate(obj.periods):
                data_all.append((obj.obj, p, obj.peak_signal_to_noise[i]))

    data_all = sorted(data_all, key=lambda x: x[1])
    data_freq = [(d[0], 1.0 / d[1], d[2]) for d in data_all]
    data_freq = sorted(data_freq, key=lambda x: x[1])

    if freq:
        return data_freq
    else:
        return data_all


def create_field_histogram(data_freq, freq_resolution):
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

    return bin_medians, binned_data


def get_aliases(freq, sampling_freq, lower_limit=None, upper_limit=None, num_aliases=3):
    aliases = []
    if lower_limit is None:
        lower_limit = freq - num_aliases * sampling_freq
    #     if lower_limit < 0:
    #         lower_limit = 0
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
            aliases.append(abs(l))
        else:
            calc_minus = False
        if u < upper_limit:
            aliases.append(u)
        else:
            calc_plus = False
        i += 1
    return aliases


def remove_aliases(bin_medians, binned_data, timeseries_length, N):
    # recursively remove aliases until max number in a bin is N.
    #     N = 2
    #     freqs = deepcopy(freq_ordered)
    bins = np.array(deepcopy(bin_medians))
    bins_data = np.array(deepcopy(binned_data))
    freqs = [x for item, x in sorted(zip(bins_data, bins), key=lambda x: len(x[0]), reverse=True)]
    c_freq = freqs[0]
    s_freq = 1. / (2. * timeseries_length)
    # s_freq = 1.0

    #     # cull periods < 1 day:
    #     freq_idx, = np.where(np.array(bins) < 1.0)
    #     bins = list(bins[freq_idx])
    #     bins_data = list(bins_data[freq_idx])

    max_num_per_bin = max([len(b) for b in bins_data])

    while max_num_per_bin > N:
        max_num_per_bin = max([len(b) for b in bins_data])
        #         print max_num_per_bin, 'largest bin size'

        aliases = get_aliases(c_freq, s_freq, num_aliases=11)

        #         fig, ax = plt.subplots(figsize=(10, 7))
        #         for bin_data in bins_data:
        #             _, freq, s2n = map(list, zip(*bin_data))
        #             ax.scatter(freq, s2n, s=1, c='b', marker='o')
        #         for alias in aliases:
        #             ax.axvline(x=alias, lw=0.5, c='r')
        #         # ax.axvline(x = freq_ordered[0], lw=0.5, c='r')
        #         ax.set_xscale('log')
        #         ax.set_ylabel('Ratio above 5 sigma threshold')
        #         ax.set_xlabel('Frequency / (1/d)')
        #         ax.set_xlim(right=10.0, left=1e-2)
        #         ax.set_yscale('log')
        #         plt.show()

        bins = [(i, b) for i, b in enumerate(bins) if not np.any(np.isclose(b, aliases, freq_resolution))]
        idxs, bins = map(list, zip(*bins))
        bins_data = list(np.array(bins_data)[idxs])

        freqs = [x for item, x in sorted(zip(bins_data, bins), key=lambda x: len(x[0]), reverse=True)]
        c_freq = freqs[0]

        max_num_per_bin = max([len(b) for b in bins_data])

    return bins_data


def find_unique_objects():
    # find unique fields and tests, and take only fields with relevant TEST.
    fieldnames = {}
    for f in os.listdir(ROOT_DIR):
        match = re.match(field_pattern, f)
        if match is not None:
            fieldnames[match.string] = []

    for fieldname in fieldnames:
        for f in os.listdir(os.path.join(ROOT_DIR, fieldname)):
            try:
                match = re.match(object_pattern, f)
                if match is not None:
                    test = match.group('test')
                    if test not in fieldnames[fieldname]:
                        fieldnames[fieldname].append(test)
            except IndexError:
                # tests.append(None)
                continue

    unique_objs = {}
    for fieldname, tests in fieldnames.iteritems():
        if TEST in tests:
            unique_objs[fieldname] = []

    print unique_objs

    # import field into memory, extract only objects of interest (once aliases have been removed)

    for fieldname in fieldnames:
        field = return_field_from_object_directory(ROOT_DIR, fieldname, TEST, silent=True)

        if field.num_objects == 0:
            continue

        field_data = create_data_tuples(field)
        bin_medians, binned_data = create_field_histogram(field_data, freq_resolution)

        timeseries_length = field.objects.values()[0].timeseries_binned[-1] - \
                            field.objects.values()[0].timeseries_binned[0]

        binned_data_pruned = remove_aliases(bin_medians, binned_data, timeseries_length, 10)

        # take only unique frequencies

        final_bins = []
        for bin_tuples in binned_data_pruned:
            if len(bin_tuples) == 1:
                final_bins.append(bin_tuples[0])

        #         print final_bins

        #         fig, ax = plt.subplots(figsize=(10, 7))
        #         for bin_data in final_bins:
        #             _, freq, s2n = bin_data
        #             ax.scatter(freq, s2n, s=1, c='b', marker='o')
        #         ax.set_xscale('log')
        #         ax.set_ylabel('Ratio above 5 sigma threshold')
        #         ax.set_xlabel('Frequency / (1/d)')
        #         ax.set_xlim(right=10.0, left=1e-2)
        # #         ax.set_yscale('log')
        #         fig.savefig('final_binned_objects_{}.png'.format(field))

        unique_objs[fieldname] = sorted(final_bins, key=lambda d: d[0])
    #     print unique_objs
    #         exit()
    with open('unique_object_info.txt', 'w') as f:
        json.dump(unique_objs, f)


def plot_unique_objects(unique_objs):
    fig, ax = plt.subplots(figsize=(10, 7))
    i = 0
    for field, final_bins in unique_objs.iteritems():
        color = color_cycle[i % len(color_cycle)]
        print field, color
        for bin_data in final_bins:
            try:
                _, freq, s2n = bin_data
                ax.scatter(freq, s2n, s=5, c=color)
            except ValueError:
                continue
        i += 1
    ax.set_xscale('log')
    ax.set_ylabel('Ratio above 5 sigma threshold')
    ax.set_xlabel('Frequency / (1/d)')
    ax.set_xlim(right=10.0, left=1e-2)
    ax.set_yscale('log')
    fig.savefig('final_binned_objects.png')


def load_unique_objects():
    with open('unique_object_info.txt', 'r') as f:
        unique_objs = json.load(f)

    for key in unique_objs.keys():
        unique_objs[key] = unique_objs[key][:]

    unique_objs = {key: unique_objs[key] for key in unique_objs.keys()[:]}

    fields = []

    for fieldname in unique_objs.keys():
        obj_periods = {}
        for d in unique_objs[fieldname]:
            if d[0] in obj_periods:
                obj_periods[d[0]][0].append(1.0 / d[1])
                obj_periods[d[0]][1].append(d[2])
            else:
                obj_periods[d[0]] = ([1.0 / d[1]], [d[2]])

        field = return_field_from_object_directory(ROOT_DIR, fieldname, TEST, silent=True, obj_ids=obj_periods.keys())

        for obj in field:
            obj.periods_ok = obj_periods[obj.obj][0]
            obj.s2n_ok = obj_periods[obj.obj][1]
        fields.append(field)

    for field in fields:
        print field, field.num_objects

    return unique_objs, fields


def load_gaia_params(field):
    xmatch_path = os.path.join(XMATCH_LOCATION.format(field.fieldname),
                               XMATCH_FILE_NAME.format(field.fieldname))
    if not os.path.exists(xmatch_path):
        print xmatch_path, 'does not exist'
        return field

    with fits.open(xmatch_path) as gfits:
        for i, obj in enumerate(gfits[1].data['Sequence_number']):
            if obj in field.objects:
                field[obj].Gaia_Teff = gfits[1].data['Gaia_Teff'][i]
                field[obj].Gaia_Radius = gfits[1].data['Gaia_Radius'][i]
                field[obj].Gaia_Lum = gfits[1].data['Gaia_Lum'][i]
                field[obj].Gaia_Parallax = gfits[1].data['Gaia_Parallax'][i]
                field[obj].TWOMASS_Hmag = gfits[1].data['2MASS_Hmag'][i]
                field[obj].TWOMASS_Kmag = gfits[1].data['2MASS_Kmag'][i]
                field[obj].APASS_Vmag = gfits[1].data['APASS_Vmag'][i]
                field[obj].APASS_Bmag = gfits[1].data['APASS_Bmag'][i]
                field[obj].Gaia_Gmag = gfits[1].data['Gaia_Gmag'][i]
                field[obj].BminusV = field[obj].APASS_Bmag - field[obj].APASS_Vmag
                field[obj].HminusK = field[obj].TWOMASS_Hmag - field[obj].TWOMASS_Kmag
                field[obj].GminusK = field[obj].Gaia_Gmag - field[obj].TWOMASS_Kmag

    return field


def generate_hr_data(fields):
    teff = []
    luminosity = []
    for field in fields:
        xmatch_path = os.path.join(XMATCH_LOCATION.format(field.fieldname),
                                   XMATCH_FILE_NAME.format(field.fieldname))
        if not os.path.exists(xmatch_path):
            print xmatch_path, 'does not exist'
            return
        with fits.open(xmatch_path) as gfits:
            teff.extend(gfits[1].data['Gaia_Teff'])
            luminosity.extend(gfits[1].data['Gaia_Lum'])
    return teff, luminosity


def generate_object_pdf(unique_objs, fields):
    #     for obj in fields[0]:
    #         print obj.GminusK

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.

    teff, luminosity = generate_hr_data(fields)

    with PdfPages('GACF_OUTPUTS.pdf') as pdf:

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        table_outline = r"""\begin{table}[]
                             \begin{tabular}{@{}ll@{}}
                             \hline
                             \textbf{NGTS Field} & \textbf{Number of Periods of Interest} \\ \hline
                             %s \hline
                             \textbf{Total} & %d \\ \hline
                             \end{tabular}
                             \end{table}""".replace('\n', ' ')

        row_outline = r"""%s     & {%d}           \\  """
        total_objs = 0
        rows = []
        for field in unique_objs:
            num_p = len(unique_objs[field])
            row = row_outline % (str(field), num_p)
            total_objs += num_p
            rows.append(row)
        rows_text = ' '.join(rows)

        text_table = (table_outline % (rows_text, total_objs)).replace('_', '\_')

        # plot overview of data on first page

        #        fig, ax = plt.subplots(figsize=(8.27, 11.69))

        page = plt.figure(figsize=(8.27, 11.69))
        height_ratios = [0.2, 0.8]
        width_ratios = [1, 1]
        gs = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=height_ratios, width_ratios=width_ratios)
        ax_info = page.add_subplot(gs[0, :])
        ax_plt = page.add_subplot(gs[1, 0])
        ax_hr = page.add_subplot(gs[1, 1])

        ax_info.axis('off')
        ax_info.text(0, 0, text_table, size=18, va='center')

        i = 0
        min_p = np.inf
        max_p = -np.inf
        for field, final_bins in unique_objs.iteritems():
            color = color_cycle[i % len(color_cycle)]
            print field, color
            for bin_data in final_bins:
                try:
                    _, freq, s2n = bin_data
                    period = 1.0 / freq
                    if period > max_p:
                        max_p = period
                    if period < min_p:
                        min_p = period
                    ax_plt.scatter(freq, s2n, s=5, c=color)
                except ValueError:
                    continue
            i += 1
        ax_plt.set_xscale('log')
        ax_plt.set_ylabel('Ratio above 5 sigma threshold')
        ax_plt.set_xlabel('Frequency / (1/d)')
        ax_plt.set_xlim(right=10.0, left=1e-2)
        ax_plt.set_yscale('log')
        ax_plt.set_ylim([1, 100])

        cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min_p, vmax=max_p)

        for field in fields:
            for obj in field:
                for period in obj.periods_ok:
                    ax_hr.scatter(obj.Gaia_Teff, obj.Gaia_Lum, s=1, c=cmap(norm(period)))

        ax_hr.set_xscale('log')
        ax_hr.set_yscale('log')
        ax_hr.set_ylabel('Gaia Bolometric Luminosity / Solar Units')
        ax_hr.set_xlabel('Gaia Effective Temperature / K')
        ax_hr.set_xlim([1e4, 3e3])
        ax_hr.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_hr.xaxis.set_minor_formatter(mticker.ScalarFormatter())

        cax, _ = mpl.colorbar.make_axes(ax_hr)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.set_label('G-ACF Period / Days')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # one page per object of interest
        num_pages = sum([f.num_objects for f in fields])
        obj_count = 1
        for field in fields:
            for obj in field:
                try:
                    Tex_Title = str(obj).replace('_', '\_')

                    radius = (float(obj.Gaia_Radius) if obj.Gaia_Radius is not None else np.nan)
                    distance = (1000. / float(obj.Gaia_Parallax) if obj.Gaia_Parallax not in [0, None] else np.nan)
                    lum = (float(obj.Gaia_Lum) if obj.Gaia_Lum is not None else np.nan)

                    TeX_Table = r"""\begin{table}[]
                                 \begin{tabular}{@{}ll@{}}
                                 \hline
                                 \textbf{Parameter} & \textbf{Value} \\ \hline
                                 NGTS Magnitude     & {%.4f}           \\
                                 Gaia $G$ Magnitude & {%.4f}           \\
                                 Gaia $T_{eff}$     & {%.4f} K           \\
                                 Gaia Luminosity     & {%.4f} $L_{sun}$            \\
                                 Gaia Radius        & {%.4f} $R_{sun}$       \\
                                 Gaia Distance      & {%.4f} pc       \\
                                 G minus K          & {%.4f}           \\ \hline
                                 \end{tabular}
                                 \end{table}""".replace('\n', ' ') % (float(np.nan), float(obj.Gaia_Gmag),
                                                                      float(obj.Gaia_Teff), lum, radius, distance,
                                                                      float(obj.GminusK))

                    TeX_periods = 'Periods: ' + ', '.join('%.2f' % p for p in obj.periods_ok) + \
                                  '\n Ratio above 5 sigma threshold: ' + \
                                  ', '.join('%.2f' % p for p in obj.s2n_ok)

                    page = plt.figure(figsize=(8.27, 11.69))
                    height_ratios = [0.5, 1, 2, 2, 2, 1, 1]
                    gs = gridspec.GridSpec(ncols=2, nrows=7, height_ratios=height_ratios)
                    ax_title = page.add_subplot(gs[0, 0])
                    ax_hr = page.add_subplot(gs[0:2, 1])
                    ax_table = page.add_subplot(gs[1, 0])
                    #                 ax_periods = page.add_subplot(gs[1, 1])
                    ax_data = page.add_subplot(gs[2, :])
                    ax_autocol = page.add_subplot(gs[3, :])
                    ax_ft = page.add_subplot(gs[4, :])
                    ax_phase1 = page.add_subplot(gs[5, 0])
                    ax_phase2 = page.add_subplot(gs[6, 0])
                    ax_phase3 = page.add_subplot(gs[5, 1])
                    ax_phase4 = page.add_subplot(gs[6, 1])

                    phase_axs = [ax_phase1, ax_phase2, ax_phase3, ax_phase4]

                    # gs = gridspec.GridSpec(5, 6)
                    # ax_table = plt.subplot(gs[3:, :])

                    ax_title.axis('off')
                    ax_table.axis('off')
                    #                ax_periods.axis('off')
                    ax_title.text(0, 0, Tex_Title, size=12, va='center')
                    ax_table.text(0, 0, TeX_Table, size=10, va='center')
                    #                ax_periods.text(0, 0, TeX_periods, size=10, va='center')

                    if lum is not None:
                        ax_hr.scatter(teff, luminosity, alpha=0.5, c='k', s=1)
                        ax_hr.scatter(obj.Gaia_Teff, lum, s=10, c='r')
                        ax_hr.set_xscale('log')
                        ax_hr.set_yscale('log')
                        ax_hr.set_ylabel('$L_{Bol}$  / $L_{Sun}$')
                        ax_hr.set_xlabel('Gaia Effective Temperature / K')
                        ax_hr.set_xlim([1e4, 3e3])
                        ax_hr.xaxis.set_major_formatter(mticker.ScalarFormatter())
                        ax_hr.xaxis.set_minor_formatter(mticker.ScalarFormatter())
                    else:
                        ax_hr.axis('off')

                    ax_data.scatter(obj.timeseries_binned, obj.flux_binned, s=0.1)
                    ax_data.set_title('Flux binned to {} mins'.format(obj.tbin))
                    ax_data.set_xlabel('Time / days')
                    ax_data.set_ylabel('Normalised Flux')

                    ax_autocol.scatter(obj.lag_timeseries, obj.correlations, s=0.5)
                    ax_autocol.set_title('Generalised Autocorrelation')
                    ax_autocol.set_xlabel('Lag / days')
                    ax_autocol.set_ylabel('G-ACF')

                    ax_ft.plot(obj.period_axis, obj.ft, '--', marker='o', ms=1, mew=2, lw=0.1)
                    # ax_ft.scatter(obj.period_axis[obj.peak_indexes], obj.ft[obj.peak_indexes], 'r+', ms=5)
                    ax_ft.set_title('FFT of Autocorrelation')
                    ax_ft.set_xlabel('Period / days')
                    ax_ft.set_ylabel('FFT Power')
                    ax_ft.set_xscale('log')
                    ax_ft.set_xlim(left=0)

                    for p in obj.periods_ok:
                        ax_ft.axvline(x=p, lw=0.5, c='r')

                    plotted = []
                    for i, ax in enumerate(phase_axs):
                        if i < len(obj.periods):
                            phase_app, data_app = utils.append_to_phase(utils.create_phase(obj.timeseries_binned,
                                                                                           obj.periods[i], 0),
                                                                        obj.flux_binned)
                            binned_phase_app, binned_data_app = utils.bin_phase_curve(phase_app, data_app)

                            ax.scatter(phase_app, data_app, s=0.1)
                            ax.scatter(binned_phase_app, binned_data_app, s=5, c='r')

                            ax.set_ylim(top=np.nanmax(binned_data_app) * 1.01, bottom=np.nanmin(binned_data_app * 0.99))

                            ax.set_title('Data phase folded on {0:.4f} day period'.format(
                                obj.periods[i]))
                            if obj.periods[i] in obj.periods_ok:
                                ax.spines['bottom'].set_color('red')
                                ax.spines['top'].set_color('red')
                                ax.spines['left'].set_color('red')
                                ax.spines['right'].set_color('red')
                            plotted.append(obj.periods[i])
                        else:
                            ax.axis('off')
                    not_plotted = [p for p in obj.periods_ok if p not in plotted]
                    if len(obj.periods) > len(obj.periods_ok) and len(not_plotted) > 0:
                        for p in not_plotted:
                            for i, pp in enumerate(reversed(plotted)):
                                if pp not in obj.periods_ok:
                                    ii = len(phase_axs) - i - 1
                                    ax = phase_axs[ii]
                                    ax.clear()

                                    phase_app, data_app = utils.append_to_phase(
                                        utils.create_phase(obj.timeseries_binned,
                                                           p, 0),
                                        obj.flux_binned)
                                    binned_phase_app, binned_data_app = utils.bin_phase_curve(phase_app, data_app)

                                    ax.scatter(phase_app, data_app, s=0.1)
                                    ax.scatter(binned_phase_app, binned_data_app, s=5, c='r')

                                    ax.set_ylim(top=np.nanmax(binned_data_app) * 1.01,
                                                bottom=np.nanmin(binned_data_app * 0.99))

                                    ax.set_title('Data phase folded on {0:.4f} day period'.format(p))

                                    ax.spines['bottom'].set_color('red')
                                    ax.spines['top'].set_color('red')
                                    ax.spines['left'].set_color('red')
                                    ax.spines['right'].set_color('red')
                                    plotted[ii] = p
                                    break

                    gs.tight_layout(page)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                    print 'Processed obj {} of {}'.format(obj_count, num_pages), "\r",
                    obj_count += 1

                except Exception:
                    print 'Failed to process obj', obj
                    continue
            #     break
            # break

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'G-ACF extracted periods'
        d['Author'] = 'Joshua Briegal'
        d['CreationDate'] = datetime.datetime.today()

        return


def mamajek_comparison():
    # get mamajek masses from g-k index
    df = pd.read_csv('/appch/data/jtb34/GitHub/GACF/example/RunonFieldSep18/mamajek_stellar_properties.csv',
                     sep=' *, *')

    df['G-Ks'] = df[u'V-Ks'] - df[u'V-G']

    mamajek_mass = df['Msun']
    mamajek_gminusk = df['G-Ks']

    idx_ok = ~np.isnan(mamajek_gminusk) & ~np.isnan(mamajek_mass)
    mamajek_mass = mamajek_mass[idx_ok]
    mamajek_gminusk = mamajek_gminusk[idx_ok]

    f_interp = interpolate.interp1d(mamajek_gminusk, mamajek_mass, fill_value='extrapolate')

    # generate plots of unique periods extracted vs gaia params

    p_rot = []
    g_minus_k = []
    t_eff = []
    radius = []

    for field in fields:
        for obj in field:
            for period in obj.periods_ok:
                p_rot.append(period)
                g_minus_k.append(obj.GminusK)
                t_eff.append(obj.Gaia_Teff)
                radius.append(obj.Gaia_Radius)

    mass = f_interp(g_minus_k)

    with PdfPages('Correlations with Gaia Params.pdf') as pdf:

        pointsize = 5
        figsize = (11.69, 8.27)

        # g minus k
        page, ax = plt.subplots(figsize=figsize)

        ax.scatter(g_minus_k, p_rot, s=pointsize)
        ax.set_xlabel('G minus K')
        ax.set_ylabel('G-ACF extracted rotation period')
        ax.set_title('Rotation period vs G-K')
        ax.set_yscale('log')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # mamajek mass derivation
        page, ax = plt.subplots(figsize=figsize)

        ax.scatter(mamajek_gminusk, mamajek_mass, c='r', label='Mamajek Values')
        ax.scatter(g_minus_k, mass, c='b', label='Interpolated Values')
        ax.legend()
        ax.set_xlabel('G minus K')
        ax.set_ylabel('Mass / MSun')
        ax.set_title('Mass vs G-K index')
        # ax.set_yscale('log')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Mamajek derived mass
        page, ax = plt.subplots(figsize=figsize)

        ax.scatter(mass, p_rot, s=pointsize)
        ax.set_xlabel('Mass / MSun')
        ax.set_ylabel('G-ACF extracted rotation period')
        ax.set_title('Rotation period vs Stellar Mass')
        ax.set_yscale('log')
        # ax.set_xlim([])

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # t_eff
        page, ax = plt.subplots(figsize=figsize)

        ax.scatter(t_eff, p_rot, s=pointsize)
        ax.set_xlabel('Effective Temperature / K')
        ax.set_ylabel('G-ACF extracted rotation period')
        ax.set_title('Rotation period vs effective temperature')
        ax.set_xlim([6500, 2800])
        ax.set_yscale('log')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # radius
        page, ax = plt.subplots(figsize=figsize)

        ax.scatter(radius, p_rot, s=pointsize)
        ax.set_xlabel('Stellar Radius / R_sun')
        ax.set_ylabel('G-ACF extracted rotation period')
        ax.set_title('Rotation period vs Stellar Radius')
        ax.set_yscale('log')
        ax.set_xscale('log')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        d = pdf.infodict()
        d['Title'] = 'G-ACF extracted periods vs Gaia Params'
        d['Author'] = 'Joshua Briegal'
        d['CreationDate'] = datetime.datetime.today()


if __name__ == '__main__':

    unique_objs, fields = load_unique_objects()

    for field in fields:
        field = load_gaia_params(field)

    mamajek_comparison()

    # generate_object_pdf(unique_objs, fields)
