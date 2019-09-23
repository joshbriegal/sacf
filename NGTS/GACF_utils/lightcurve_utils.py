import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter
from astropy.time import Time
from astropy.coordinates import get_moon, get_sun
from .ngtsio_utils import NGTS_EPOCH, medsig

def create_phase(time_series, period, epoch):
    return np.mod(np.array(time_series) - epoch, period) / period


def rescale_phase(phase, offset=0.2):
    return [p - 1 if p > 1 - offset else p for p in phase]


def append_to_phase(phase, data, amt=0.05):
    indexes_before = [i for i, p in enumerate(phase) if p > 1 - amt]
    indexes_after = [i for i, p in enumerate(phase) if p < amt]

    phase_before = [phase[i] - 1 for i in indexes_before]
    data_before = [data[i] for i in indexes_before]

    phase_after = [phase[i] + 1 for i in indexes_after]
    data_after = [data[i] for i in indexes_after]

    return np.concatenate((phase_before, phase, phase_after)), \
        np.concatenate((data_before, data, data_after))


def bin_phase_curve(phase, data, statistic='median', bins=20):
    bin_medians, bin_edges, bin_number = binned_statistic(
        phase, data, statistic=statistic, bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    return bin_centers, bin_medians


def remove_transit(time_series, data, period, epoch, plot=False):
    # phase fold
    phase = create_phase(time_series, period, epoch)
    epoch_phase = np.mod(0, period) / period
    rescaled_phase = rescale_phase(phase)

    phase_width = 0.04

    transit_phase = [p for p in rescaled_phase if -
                     phase_width < p < phase_width]
    transit_indexes = [p[0] for p in enumerate(
        rescaled_phase) if -phase_width < p[1] < phase_width]
    transit_data = [data[i] for i in transit_indexes]
    transit_times = [time_series[i] for i in transit_indexes]

    if plot:
        plt.scatter(rescaled_phase, data, s=0.1)
        plt.scatter(transit_phase, transit_data, s=0.1)
        plt.axvline(x=epoch_phase)
        plt.show()

    non_transit_phase = [rescaled_phase[i] for i in range(
        0, len(rescaled_phase)) if i not in transit_indexes]
    non_transit_data = [data[i]
                        for i in range(0, len(data)) if i not in transit_indexes]
    non_transit_times = [time_series[i] for i in range(
        0, len(time_series)) if i not in transit_indexes]
    if plot:
        plt.scatter(non_transit_phase, non_transit_data, s=0.1)
        plt.show()

    return non_transit_times, non_transit_data


def normalise_data_by_polynomial(time_series, data, order=3, plot=False, filename=None):
    poly_fit = np.poly1d(np.polyfit(time_series, data, order))
    data_fit = [poly_fit(t) for t in time_series]
    norm_data = np.divide(np.array(data), np.array(data_fit))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(time_series, data, s=0.1, label='Data', alpha=0.5)
        ax.plot(time_series, data_fit, c='k', label=str(poly_fit))
        ax.scatter(time_series, norm_data, s=0.1, label='Normalised Data')

        ax.set_title('Data with polynomial fit')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalised Intensity')

        ax.legend(fontsize=8)

        fig.savefig(filename + '/data_normalisation.pdf')

        plt.close(fig)

    return norm_data


def normalise_data_by_spline(time_series, data, plot=False, filename=None, **kwargs):
    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(time_series, data, **kwargs)
    data_fit = spl(time_series)
    norm_data = np.divide(np.array(data), np.array(data_fit))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(time_series, data, s=0.1, label='Data', alpha=0.5)
        ax.plot(time_series, data_fit, c='k', label='spline fit')
        ax.scatter(time_series, norm_data, s=0.1, label='Normalised Data')

        ax.set_title('Data with spline fit')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalised Intensity')

        ax.legend(fontsize=8)

        fig.savefig(filename + '/data_normalisation.pdf')

        plt.close(fig)

    return norm_data


def remove_moon_signal(t, f, ferr, P, epoch=0):
    """
    determine period and phase to fold on
    P = estimate from LS fits
    phase = same for all LCs - epoch = new moon works best, need to calculate separately
    try:
        savitzky_golay(y, 51, 3) # window size 51, polynomial order 3
        or moving average from pandas
    subtract from f
    return f
    """
    medf = np.median(f)
    f = f - medf
    t2r, f2r, ferr2r = t, f, ferr

    # get flux in phase-order

    phase = np.mod(t - epoch, P) / P
    pidx = np.argsort(phase)

    tp = t[pidx]
    fp = f[pidx]
    fperr = ferr[pidx]

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # detrend for 'moon' signal: uses two-step process
    # first scipy.signal.savgol_filter() followed by convolution (smooth())
    # window size 151, polynomial order 1
    sgf = savgol_filter(fp, 151, 1, mode='wrap')
    sgf = smooth(sgf, 20)

    # re-order in time
    tidx = np.argsort(t)
    f2r = fp[tidx] - sgf[tidx]
    t2r = tp[tidx]
    ferr2r = fperr[tidx]

    # double-check ordering
    idx = np.argsort(t2r)
    t2r = t2r[idx]
    f2r = f2r[idx] + medf
    ferr2r = ferr2r[idx]

    if any(t2r[1:] - t2r[:-1] < 0.):
        raise ValueError(
            "Re-ordering arrays back into time-order has failed somehow...")

    return t2r, f2r, ferr2r, sgf[tidx][idx] + medf

def split_and_compute_percentile_per_specified_time_period(t, f, tref, dt, percentiles=[10., 90.], nsigma=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.array(t)
        f = np.array(f)
        fpn=[]
        treg = np.concatenate([ np.r_[tref  : t.min() : -dt][::-1],
                                np.r_[tref+dt : t.max() :  dt] ])
        nreg = len(treg)
        freg = np.zeros(nreg) + np.nan
        for j in np.arange(nreg):
            l = np.logical_and( (t >= treg[j]), (t < treg[j]+dt) )
            if l.any():
                # compute
                fsl = f[l]
                if nsigma:
                    med, sig = medsig(fsl)
                    fsl[abs(med - fsl) > abs(nsigma * sig)] = np.nan # sigma clip
                perc = np.nanpercentile(fsl, percentiles)
                fpn.append(perc[1]-perc[0])
        fpn = np.array(fpn)
    #     spread = np.median(fpn[:,1]-fpn[:,0])
    #     spread = np.median(np.diff(fpn))
        spread = np.nanmedian(fpn)
    return spread

def split_and_compute_percentile_per_specified_time_period(t, f, tref, dt,
                                                           percentiles=[10., 90.], nsigma=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.array(t)
        f = np.array(f)
        fpn = []
        treg = np.concatenate([np.r_[tref: t.min(): -dt][::-1],
                               np.r_[tref + dt: t.max():  dt]])
        nreg = len(treg)
        if nreg == 0 and (np.min(t) + dt > np.max(t)):
            # take whole series as input
            treg = [np.min(t)]
            nreg = 1
        freg = np.zeros(nreg) + np.nan
        for j in np.arange(nreg):
            l = np.logical_and((t >= treg[j]), (t < treg[j] + dt))
            if l.any():
                # compute
                fsl = f[l]
                if nsigma:
                    med, sig = medsig(fsl)
                    fsl[abs(med - fsl) > abs(nsigma * sig)
                        ] = np.nan  # sigma clip
                perc = np.nanpercentile(fsl, percentiles)
                fpn.append(perc[1] - perc[0])
        fpn = np.array(fpn)
    #     spread = np.median(fpn[:,1]-fpn[:,0])
    #     spread = np.median(np.diff(fpn))
        spread = np.nanmedian(fpn)
    return spread


def moon_phase_angle(time, ephemeris=None):
    """
    Calculate lunar orbital phase in radians.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time of observation

    ephemeris : str, optional
        Ephemeris to use.  If not given, use the one set with
        `~astropy.coordinates.solar_system_ephemeris` (which is
        set to 'builtin' by default).

    Returns
    -------
    i : float
        Phase angle of the moon [radians]
    """
    # TODO: cache these sun/moon SkyCoord objects

    sun = get_sun(time)
    moon = get_moon(time, ephemeris=ephemeris)
    elongation = sun.separation(moon)
    return np.arctan2(sun.distance * np.sin(elongation),
                      moon.distance - sun.distance * np.cos(elongation))


def moon_illumination(time, ephemeris=None):
    """
    Calculate fraction of the moon illuminated.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time of observation

    ephemeris : str, optional
        Ephemeris to use.  If not given, use the one set with
        `~astropy.coordinates.solar_system_ephemeris` (which is
        set to 'builtin' by default).

    Returns
    -------
    k : float
        Fraction of moon illuminated
    """
    i = moon_phase_angle(time, ephemeris=ephemeris)
    k = (1 + np.cos(i)) / 2.0
    return k.value


def get_new_moon_epoch(timeseries):
    med_time = np.median(timeseries)
    moon_time = np.linspace(med_time - 30, med_time + 30, 100)
    moons = [moon_illumination(Time(t + NGTS_EPOCH, format='jd'))
             for t in moon_time]
    moon_epoch = Time(moon_time[np.argmin(moons)], format='jd').jd
#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.plot(moon_time, moons)
#     ax.axvline(x=moon_time[np.argmin(moons)])
    return moon_epoch, moon_time, moons


def split_phase(phase, data, timeseries=None, buff=0.9, nperiods=1):
    # returns list of lists of data & phases for complete periods
    # (requires sorted phased timeseries)
    # buff => require this much phase coverage in first and last segments
    phases = []
    datas = []
    timeseriess = [] if timeseries is not None else None

    idx_changes = np.where(np.diff(phase) < 0)[0][::nperiods]
    use_first = True if (phase[0] < 1.0 - buff) else False
    use_last = True if (phase[-1] > buff) else False

    if use_first:
        phases.append(phase[:idx_changes[0]])
        datas.append(data[:idx_changes[0]])
        if timeseriess is not None:
            timeseriess.append(timeseries[:idx_changes[0]])

    for i, idx in enumerate(idx_changes[:-1]):
        phases.append(phase[idx + 1:idx_changes[i + 1]])
        datas.append(data[idx + 1:idx_changes[i + 1]])
        if timeseriess is not None:
            timeseriess.append(timeseries[idx + 1:idx_changes[i + 1]])

    if use_last or np.any(np.diff(phase[idx_changes[-1] + 1:]) < 0):
        phases.append(phase[idx_changes[-1]:])
        datas.append(data[idx_changes[-1]:])
        if timeseriess is not None:
            timeseriess.append(timeseries[idx_changes[-1]:])
    if timeseriess is not None:
        return phases, datas, timeseriess
    else:
        return phases, datas
#     save = False
#     phase_count = 0
#     for p, d in zip(phase, data):
