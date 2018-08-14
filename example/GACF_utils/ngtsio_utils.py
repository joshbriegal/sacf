import numpy as np
import scipy as sp
import warnings

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


def get_ngts_data(fieldname, obj_id, ngts_version, nsig2keep=None, do_relflux=True, keys=None, silent=True,
                  return_dic=False, check_num_data=True, bad_data_ratio_threshold=0.5):
    if keys is None:
        keys = ['HJD', 'SYSREM_FLUX3', 'FLAGS']
    dic = ngtsio.get(fieldname=fieldname, keys=keys, obj_id=obj_id,
                     ngts_version=ngts_version, silent=silent)
    if check_num_data:
        if 'SYSREM_FLUX3' in keys:
            if dic['SYSREM_FLUX3'].size > 1:
                try:
                    idx_ok = np.fromiter((is_good_data(flux, bad_data_ratio_threshold)
                                          for flux in dic['SYSREM_FLUX3']), dtype=bool)
                    for key in keys:
                        try:
                            dic[key] = dic[key][idx_ok]
                        except (TypeError, IndexError, ValueError):
                            continue
                except (TypeError, IndexError, ValueError) as e:
                    pass

    if not dic:
        return None
    else:
        if return_dic:
            return dic
        else:
            return clean_ngts_data(dic, nsig2keep=nsig2keep, do_relflux=do_relflux)


def clean_ngts_data(dic, nsig2keep=None, do_relflux=True, bad_data_ratio_threshold=0.5):
    dic['SYSREM_FLUX3'][np.where(dic['FLAGS'] != 0.)] = np.nan
    flux = dic['SYSREM_FLUX3']
    num_observations_initial = len(flux)
    if nsig2keep is not None:
        med, sig = medsig(flux)
        idx_ok = (abs(flux - med) < nsig2keep * sig) * (flux != 0.)
    else:
        idx_ok = (flux != 0.)
    timeseries = dic['HJD'][idx_ok] / TIME_CONVERSIONS['d2s']
    flux = flux[idx_ok]
    median_flux, flux_std_dev = medsig(flux)
    if do_relflux:
        flux /= median_flux
    num_observations = len(timeseries)
    try:
        is_ok = True if float(num_observations) / float(
            num_observations_initial) > bad_data_ratio_threshold else False
    except ZeroDivisionError:
        is_ok = False
    return flux, timeseries, median_flux, flux_std_dev, num_observations, is_ok


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
