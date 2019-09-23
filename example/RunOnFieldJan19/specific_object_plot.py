import sys
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/appch/data/jtb34/GitHub/GACF/')
from NGTS.NGTS_Field import return_field_from_object_directory
from NGTS.NGTS_Object import return_object_from_json_string
from NGTS.GACF_utils import TIME_CONVERSIONS, get_ngts_data, rebin_err_chunks

from ngtsio import ngtsio
from astropy.io import fits
from GACF import find_correlation_from_lists_cpp
from NGTS.GACF_utils import fourier_transform_and_peaks
import random

plt.rcParams.update({'font.size': 14})


def get_json_string(fieldname, obj_id):
    object_file_name = '{0}_VERSION_{1}'.format(obj_id, CYCLE)
    object_file = os.path.join(*[ROOT_DIR, fieldname, object_file_name, 'object.json'])
    with open(object_file, 'r') as f:
        jsonstr = f.read()
    return jsonstr


def get_gaia_params(obj):
    XMATCH_LOCATION = '/appch/data/jtb34/{}/cross_match/'.format(obj.field)
    FILE_NAME = 'Uncut_Final_{}.fits'.format(obj.field)
    xmatch_data = fits.open(os.path.join(XMATCH_LOCATION, FILE_NAME))

    for i, obj_id in enumerate(xmatch_data[1].data['Sequence_number']):
        if int(obj.obj) == int(obj_id):
            obj.Gaia_Teff = xmatch_data[1].data['Gaia_Teff'][i]
            obj.Gaia_Radius = xmatch_data[1].data['Gaia_Radius'][i]
            obj.Gaia_Lum = xmatch_data[1].data['Gaia_Lum'][i]
            obj.TWOMASS_Hmag = xmatch_data[1].data['2MASS_Hmag'][i]
            obj.TWOMASS_Kmag = xmatch_data[1].data['2MASS_Kmag'][i]
            obj.APASS_Vmag = xmatch_data[1].data['APASS_Vmag'][i]
            obj.APASS_Bmag = xmatch_data[1].data['APASS_Bmag'][i]
            obj.Gaia_Gmag = xmatch_data[1].data['Gaia_Gmag'][i]
            obj.BminusV = obj.APASS_Bmag - obj.APASS_Vmag
            obj.HminusK = obj.TWOMASS_Hmag - obj.TWOMASS_Kmag
            obj.GminusK = obj.Gaia_Gmag - obj.TWOMASS_Kmag
    return obj


def create_obj(fieldname, obj_id):
    obj = return_object_from_json_string(get_json_string(fieldname, obj_id))
    return get_gaia_params(obj)


def get_sky_bkg(obj):
    dic = ngtsio.get(fieldname=obj.field, keys=['OBJ_ID', 'SKYBKG', 'HJD'], obj_id=obj.obj,
                     ngts_version=obj.test, silent=False)

    idx_ok = ~np.isnan(dic['SKYBKG'])
    timeseries_binned, flux_binned, _ = \
        rebin_err_chunks(dic['HJD'][idx_ok] * TIME_CONVERSIONS['s2d'], dic['SKYBKG'][idx_ok],
                         dt=20 * TIME_CONVERSIONS['m2d'],
                         get_err_on_mean=False)
    obj.sky_bkg = flux_binned
    obj.sky_bkg_hjd = timeseries_binned
    return obj


def get_obj_ids(fieldname, test):
    dic = ngtsio.get(fieldname=fieldname, keys=['OBJ_ID'], ngts_version=test, silent=True)
    return map(int, dic['OBJ_ID'])


if __name__ == '__main__':
    ROOT_DIR = '/home/jtb34/rds/hpc-work/GACF_OUTPUTS'
    CYCLE = 'CYCLE1807'
    FIELDNAME = 'NG2346-BLAH'

    obj_ids = get_obj_ids(FIELDNAME, CYCLE)
    # print len(obj_ids)
    # print obj_ids[:5]

    obj_id = obj_ids[random.randint(0, len(obj_ids) - 1)]

    # NGTS Object NG0442-3345_17369 (CYCLE1802) 17.85 sigma times threshold

    obj = create_obj(FIELDNAME, obj_id)
    # print obj.Gaia_Gmag
    obj = get_sky_bkg(obj)

    obj.correlations = None
    obj.lag_timeseries = None
    
    import time

    start=time.time()
    obj.calculate_periods_from_autocorrelation()
    end = time.time()

    print 'time taken', end-start, 'seconds'

    print 'Sky background saved'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))

    # ax1.set_title('Data')
    ax1.scatter(obj.timeseries_binned, obj.flux_binned, s=0.5)
    ax1.set_xlabel('HJD')
    ax1.set_ylabel('Normalised Flux')

    ax2.set_title('G-ACF')
    ax2.scatter(obj.lag_timeseries, obj.correlations, s=0.5)
    ax2.set_xlabel('Lag / days')
    ax2.axvline(x=obj.periods[0], c='r', lw=2)
    ax2.set_ylabel('G-ACF')

    ax3.set_title('FFT')
    ax3.plot(obj.period_axis, obj.ft, lw=2)
    ax3.scatter(obj.period_axis[obj.peak_indexes], obj.ft[obj.peak_indexes], s=40, c='r', marker='+')
    ax3.set_ylabel('FFT Power')
    ax3.set_xscale('log')
    ax3.set_xlim([0, 100])
    ax3.set_xlabel('Period / days')
    ax3.axvline(x=obj.periods[0], c='r', lw=2)

    fig.suptitle(str(obj) + ' Gaia Gmag {}'.format(obj.Gaia_Gmag))
    fig.tight_layout()

    plt.savefig('autocorrelation_{}.pdf'.format(obj))

    print 'Data plot created'
    print 'Periods: {}'.format(obj.periods)
    # SKY BACKGROUND
    
    lag_timeseries, correlations, _ = find_correlation_from_lists_cpp(obj.sky_bkg_hjd, obj.sky_bkg,
                                                                      lag_resolution=20 * TIME_CONVERSIONS['m2d'])
    ft, period_axis, indexes = fourier_transform_and_peaks(correlations, lag_timeseries)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))

    # ax1.set_title('Data')
    ax1.scatter(obj.sky_bkg_hjd, obj.sky_bkg, s=0.5)
    ax1.set_xlabel('HJD')
    ax1.set_ylabel('Normalised Flux')

    ax2.set_title('G-ACF')
    ax2.scatter(lag_timeseries, correlations, s=0.5)
    ax2.set_xlabel('Lag / days')
    ax2.axvline(x=period_axis[indexes[0]], c='r', lw=2)
    ax2.set_ylabel('G-ACF')
    ax2.set_xlim(left=0)

    ax3.set_title('FFT')
    ax3.plot(period_axis, ft, lw=2)
    ax3.scatter(period_axis[indexes], ft[indexes], s=40, c='r', marker='+')
    ax3.set_ylabel('FFT Power')
    ax3.set_xscale('log')
    ax3.set_xlim([0, 100])
    ax3.set_xlabel('Period / days')
    ax3.axvline(x=period_axis[indexes[0]], c='r', lw=2)

    fig.suptitle(str(obj) + "SKY BACKGROUND")
    fig.tight_layout()

    plt.savefig('autocorrelation_{}_SKY_BKG.pdf'.format(obj))
    print 'Background periods : {}'.format(period_axis[indexes])
