import numpy as np
import scipy as sp
import warnings
import itertools
import traceback

try:
    from ngtsio import ngtsio
except ImportError:
    warnings.warn("ngtsio not imported")

TIME_CONVERSIONS = dict(d2s=24. * 60. * 60.,
                        d2m=24. * 60.,
                        d2h=24.,
                        h2m=60.,
                        m2s=60.,
                        s2d=1. / (24. * 60. * 60.),
                        m2d=1. / (24. * 60.),
                        h2d=1. / 24.)

NGTS_EPOCH = 2456658.5


def medsig(a):
    """Compute median and MAD-estimated scatter of array a"""
    med = sp.nanmedian(a)
    sig = 1.48 * sp.nanmedian(abs(a - med))
    return med, sig


def is_good_data(flux, bad_data_ratio_threshold=0.5):
    num_observations_total = float(len(flux))
    num_observations_good = float(len(flux[np.where(flux != 0.)]))
    if num_observations_good / num_observations_total < bad_data_ratio_threshold:
        return False
    else:
        return True


def clean_data_dict(dic, check_num_data=True, bad_data_ratio_threshold=0.5):
    keys = ['OBJ_ID', 'HJD', 'SYSREM_FLUX3', 'FLAGS']
    if 'SYSREM_FLUX3' in keys:
        if dic['SYSREM_FLUX3'].size > 1:
            try:
                idx_ok = np.fromiter((is_good_data(flux, bad_data_ratio_threshold)
                                      for flux in dic['SYSREM_FLUX3']), dtype=bool)
                for key in keys:
                    try:
                        dic[key] = np.array(dic[key])[idx_ok]
                    except (TypeError, IndexError, ValueError) as e:
                        print key, 'Key not removed properly'
                        traceback.print_exc()
            except (TypeError, IndexError, ValueError) as e:
                pass
    return dic


def pop_dict_item(dic):
    item_to_return = {}
    for key in dic.keys():
        item = dic[key]
        if type(item) is np.ndarray:
            try:
                item, rest_of_arr = dic[key][0], dic[key][1:]
                item_to_return[key] = np.array(item)
                dic[key] = rest_of_arr
            except IndexError:
                item_to_return = None
                break
        else:
            item_to_return[key] = item
    return item_to_return, dic


def get_ngts_data(fieldname, obj_id, ngts_version, nsig2keep=None, do_relflux=True, keys=None, silent=True,
                  return_dic=False, check_num_data=True, bad_data_ratio_threshold=0.5):
    if keys is None:
        keys = ['HJD', 'SYSREM_FLUX3', 'FLAGS']
    dic = ngtsio.get(fieldname=fieldname, keys=keys, obj_id=obj_id,
                     ngts_version=ngts_version, silent=silent)
    if check_num_data and dic:
        dic = clean_data_dict(dic, check_num_data, bad_data_ratio_threshold)
    if not dic:
        if return_dic:
            return None
        else:
            return None, None, None, None, None, None
    else:
        if return_dic:
            return dic
        else:
            return clean_ngts_data(dic, nsig2keep=nsig2keep, do_relflux=do_relflux)


def clean_ngts_data(dic, nsig2keep=None, do_relflux=True, bad_data_ratio_threshold=0.2, min_points=500):
    is_ok = True
    dic['SYSREM_FLUX3'][np.where(dic['FLAGS'] != 0.)] = np.nan
    message = ''
    flux = dic['SYSREM_FLUX3']
    #     print 'Flagged', len([d for d in dic['FLAGS'] if d!= 0.]), 'of', len(flux)
    if flux is None or (len(flux) < min_points):
        message += 'Not enough data ({} points unbinned) \n'.format(len(flux))
        is_ok = False
        return flux, None, None, None, None, is_ok, message
    num_observations_initial = len(flux)
    if nsig2keep is not None:
        med, sig = medsig(flux)
        with np.errstate(divide='ignore', invalid='ignore'):
            idx_ok = (abs(flux - med) < (nsig2keep * sig)) & (flux != 0.) & ~np.isnan(flux)
    else:
        idx_ok = (flux != 0.) & ~np.isnan(flux)
    timeseries = dic['HJD'][idx_ok] / TIME_CONVERSIONS['d2s']
    flux = flux[idx_ok]
    median_flux, flux_std_dev = medsig(flux)
    if do_relflux:
        flux /= median_flux
    num_observations = len(timeseries)
    try:
        ratio = (float(num_observations) / float(num_observations_initial))
        if ratio < bad_data_ratio_threshold:
            is_ok = False
            message += '{:0.2f}% of data removed due to flags / zero values'.format((1 - ratio) * 100)
    #         print ratio, is_ok
    except ZeroDivisionError:
        is_ok = False
    return flux, timeseries, median_flux, flux_std_dev, num_observations, is_ok, message


def rebin_err(t, f, dt=0.02, get_err_on_mean=False, bin_about=None):
    """
    Rebin a time-series with errors on the data (y-points).
    Apply unweighted average: ignore errors on the data to be binned and
                              perform a simple MAD estimation of the error
                              from the scatter of points
    """
    treg = np.r_[t.min():t.max():dt]
    if bin_about is not None:
        treg = np.concatenate([np.r_[bin_about: t.min(): -dt][::-1],
                               np.r_[bin_about + dt: t.max():  dt]])
    nreg = len(treg)
    freg = np.zeros(nreg) + np.nan
    freg_err = np.zeros(nreg) + np.nan
    for i in np.arange(nreg):
        l = (t >= treg[i]) * (t < treg[i] + dt)
        if l.any():
            treg[i] = np.nanmean(t[l])
            freg[i], freg_err[i] = medsig(f[l])
            if get_err_on_mean:
                # freg_err[i] /= float(len(f[l]))
                freg_err[i] /= np.sqrt(float(len(f[l])))
    l = np.isfinite(freg)
    return treg[l], freg[l], freg_err[l]


def segment_times(timeseries, max_gap):
    """
    Returns an N-D array where each row represents a separate segmentation of continuous data with no gaps
    greater than the max gap.
    """
    time_segments = []
    is_contiguous = False
    arr_n = -1
    for i, t in enumerate(timeseries):
        if not is_contiguous:
            time_segments.append([t])
            arr_n += 1
        else:
            time_segments[arr_n].append(t)
        if i + 1 < len(timeseries):
            is_contiguous = (timeseries[i + 1] - t) < max_gap
    return time_segments


def match_dimensions(ndarray, onedarray):
    """
    Return an N-D array of shape ndarray.shape with the values of onedarray
    """
    ndreturn = []
    idx = 0
    for sublist in ndarray:
        num = len(sublist)
        subreturn = onedarray[idx: idx + num]
        idx += num
        ndreturn.append(subreturn)
    return ndreturn


def rebin_err_chunks(t, f, dt=0.02, get_err_on_mean=False):
    times = segment_times(t, dt)
    fluxes = match_dimensions(times, f)
    times_binned = []
    fluxes_binned = []
    flux_errs_binned = []
    nreg = 0
    n_data = 0
    for i, each_time in enumerate(times):
        tbin, fbin, ferr = rebin_err(np.array(each_time), np.array(fluxes[i]),
                                     dt=dt, get_err_on_mean=get_err_on_mean)
        times_binned.append(tbin)
        fluxes_binned.append(fbin)
        flux_errs_binned.append(ferr)
    treg = list(itertools.chain(*times_binned))
    freg = list(itertools.chain(*fluxes_binned))
    freg_err = list(itertools.chain(*flux_errs_binned))

    return treg, freg, freg_err
