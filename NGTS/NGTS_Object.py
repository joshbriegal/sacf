"""
PyObject per NGTS object, containing light curve and associated analysis
"""
import numpy as np
from GACF import find_correlation_from_lists_cpp, GACF_LOG_MESSAGE
import GACF_utils as utils
import os
import pickle
import json
import logging
from copy import deepcopy
from astropy.time import Time


def return_object_from_json_string(json_str):
    if isinstance(json_str, str):
        dic = json.loads(json_str, cls=utils.NGTSObjectDecoder)
    elif isinstance(json_str, dict):
        dic = json_str
    else:
        raise TypeError("Input must be string or dict")
    tobj = NGTSObject(1, 1, 1)  # temp
    odic = tobj.__dict__
    for k, v in odic.iteritems():
        if k not in dic:
            # update object to include initialised values which aren't saved.
            dic[k] = v
    tobj.__dict__ = dic
    try:
        tobj.create_logger(append=True)  # JSON has no logger info
    except (IOError, AttributeError):  # file not found, don't create logger
        tobj.logger = None
    return tobj


class NGTSObject(object):

    def __init__(self, obj, field, test, lc_data2get=None, tbin=20, nsig2keep=None, do_relflux=True, min_lag=None,
                 max_lag=None, lag_resolution=None, selection_function='natural', weight_function='gaussian',
                 alpha=None, field_filename=None, logger=None):

        if field_filename is None:
            self.filename = os.getcwd()
        else:
            self.filename = os.path.join(
                field_filename, "{}_VERSION_{}".format(obj, test))

        # NGTSIO inputs
        self.obj = obj
        self.field = field
        self.test = test
        self.lc_data2get = lc_data2get if lc_data2get is not None else [
            'HJD', 'SYSREM_FLUX3', 'FLAGS']
        self.tbin = tbin
        self.nsig2keep = nsig2keep
        self.do_relflux = do_relflux

        # GACF settings
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.lag_resolution = lag_resolution
        self.selection_function = selection_function
        self.weight_function = weight_function
        self.alpha = alpha

        # NGTSIO outputs
        self.flux = None
        self.timeseries = None
        self.median_flux = None
        self.flux_std_dev = None
        self.num_observations = None

        # Binned NGTSIO outputs
        self.flux_binned = None
        self.timeseries_binned = None
        self.flux_binned_err = None
        self.median_flux_binned = None
        self.flux_binned_std_dev = None
        self.num_observations_binned = None

        # Noise threshold calculations
        self.running_noise_threshold = None
        self.gen_data_dict, self.gen_autocol_dict, self.gen_ft_dict = None, None, None

        # Autocorrelation outputs
        self.correlations, self.lag_timeseries = None, None

        # Period finding outputs
        self.ft, self.period_axis, self.periods = None, None, None
        self.peak_indexes, self.peak_percentages, self.peak_signal_to_noise = None, None, None

        self.logger = None
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        self.ok = True
        self.message = ''

    def __str__(self):
        return "NGTS Object " + str(self.field) + "_" + str(self.obj) + " ({})".format(self.test)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        elif isinstance(other, str):
            return self.obj == other
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def create_logger(self, logger=None, append=False):
        if logger is None:
            mode = 'a' if append else 'w'
            fh = logging.FileHandler(os.path.join(
                self.filename, 'peaks.log'), mode)
            fh.setFormatter(logging.Formatter())  # default formatter
            logger = logging.getLogger(str(self))
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)
        self.logger = logger

    def remove_logger(self):
        if self.logger is None:
            return
        else:
            handlers = self.logger.handlers[:]
            for handler in handlers:
                handler.close()
                self.logger.removeHandler(handler)
            self.logger = None
        return

    def safe_log(self, message):
        # does not break if logger is none
        if self.logger:
            self.logger.info(message)

    def get_binned_data(self, delete_unbinned=True):
        self.get_data()
        self.bin_data(delete_unbinned=delete_unbinned)
        return

    def get_data_from_dict(self, dic, nsig2keep=5,
                           do_relflux=True, min_points=500):  # if getting data using multiple lcs
        if not isinstance(dic["OBJ_ID"], list):
            self.flux, self.timeseries, self.median_flux, self.flux_std_dev, self.num_observations, self.ok, message = \
                utils.clean_ngts_data(
                    dic, nsig2keep=nsig2keep, do_relflux=do_relflux, min_points=min_points)
            self.message += message
            return
        else:
            idx, = np.where(dic["OBJ_ID"] == self)
            if idx.size == 0:
                self.ok = False
                return
            if len(idx) > 1:
                raise ValueError(
                    "more than one object returned with this ID - check inputs")
            new_dic = {}
            for key in dic.keys():
                try:
                    new_dic[key] = dic[key][idx]
                    if len(new_dic[key]) == 1:
                        new_dic[key] = new_dic[key][0]
                except TypeError:
                    new_dic[key] = dic[key]
                    self.flux, self.timeseries, self.median_flux, self.flux_std_dev, self.num_observations, self.ok, message = \
                        utils.clean_ngts_data(
                            new_dic, nsig2keep=nsig2keep, do_relflux=do_relflux)
                    self.message += message
                    return

    def get_data(self):
        self.flux, self.timeseries, self.median_flux, self.flux_std_dev, self.num_observations, self.ok = \
            utils.get_ngts_data(fieldname=self.field, keys=self.lc_data2get, obj_id=self.obj, ngts_version=self.test,
                                nsig2keep=self.nsig2keep, do_relflux=self.do_relflux)

        return

    def bin_data(self, delete_unbinned=True):
        if not self.ok:
            return
        self.timeseries_binned, self.flux_binned, self.flux_binned_err = \
            utils.rebin_err_chunks(self.timeseries, self.flux,
                                   dt=self.tbin * utils.TIME_CONVERSIONS['m2d'], get_err_on_mean=False)
        self.median_flux_binned, self.flux_binned_std_dev = utils.medsig(
            self.flux_binned)
        self.num_observations_binned = len(self.timeseries_binned)
        if delete_unbinned:  # should save space as we don't need unbinned data after binning
            self.flux = None
            self.timeseries = None
            self.median_flux = None
            self.flux_std_dev = None
            self.num_observations = None
        return

    def calculate_noise_threshold(self, sigma_level=5, sigma_level_calc=3):
        if self.correlations is None or self.lag_timeseries is None:
            self.calculate_autocorrelation()
        if self.flux_binned_err is None:
            if self.flux is not None and self.timeseries is not None:
                self.bin_data()
            else:
                self.get_binned_data()

        no_samples = utils.sigma_level_to_samples(sigma_level=sigma_level_calc)

        generated_data = utils.generate_random_lightcurves(
            self.timeseries_binned, self.flux_binned_err, no_samples)

        self.running_noise_threshold, self.gen_data_dict, self.gen_autocol_dict, self.gen_ft_dict = \
            utils.calculate_noise_level(time_series=self.timeseries_binned, generated_data=generated_data,
                                        sigma_level_calc=sigma_level_calc, sigma_level_out=sigma_level,
                                        min_lag=self.min_lag, max_lag=self.max_lag,
                                        lag_resolution=self.lag_resolution, alpha=self.alpha,
                                        logger=None)

        return

    def calculate_autocorrelation(self, use_binned_data=True):
        #        if not os.path.exists(self.filename):
        #            os.makedirs(self.filename)
        if self.logger is None:
            self.create_logger()
        if use_binned_data and (self.timeseries_binned is None or self.flux_binned is None):
            if self.timeseries is not None and self.flux is not None:
                self.bin_data()
            else:
                raise ValueError('No data for {}'.format(self))
#                self.get_binned_data()
        elif not use_binned_data and (self.flux is None or self.timeseries is None):
            self.get_data()

        if use_binned_data:
            (self.lag_timeseries,
             self.correlations,
             corr) = find_correlation_from_lists_cpp(self.timeseries_binned,
                                                     self.flux_binned,
                                                     min_lag=self.min_lag,
                                                     max_lag=self.max_lag,
                                                     lag_resolution=self.lag_resolution,
                                                     alpha=self.alpha)
        else:
            (self.lag_timeseries,
             self.correlations,
             corr) = find_correlation_from_lists_cpp(self.timeseries, self.flux,
                                                     min_lag=self.min_lag,
                                                     max_lag=self.max_lag,
                                                     lag_resolution=self.lag_resolution,
                                                     alpha=self.alpha)

        self.safe_log(
            GACF_LOG_MESSAGE.format(no_data=self.num_observations_binned if use_binned_data else self.num_observations,
                                    no_lag_points=len(self.lag_timeseries),
                                    lag_resolution=corr.lag_resolution))
        self.min_lag = corr.min_lag
        self.max_lag = corr.max_lag
        self.lag_resolution = corr.lag_resolution
        self.alpha = corr.alpha
        return

    def calculate_periods_from_autocorrelation(self, running_noise_threshold=None, calculate_noise_threshold=False,
                                               num_periods=None, use_binned_data=True, pmin=None, pmax=None, **kwargs):
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        if running_noise_threshold is None:
            if calculate_noise_threshold:
                self.calculate_noise_threshold()
            running_noise_threshold = self.running_noise_threshold

        if self.correlations is None or self.lag_timeseries is None:
            self.calculate_autocorrelation(use_binned_data=use_binned_data)

        time_series = self.timeseries_binned if use_binned_data else self.timeseries
        signal_to_noise = None

        ft, periods, indexes = utils.fourier_transform_and_peaks(
            self.correlations, self.lag_timeseries, n=128, **kwargs)

        # pruning: user input pmin and pmax, no periods greater than half the
        # length of the time series.
        indexes = [index for index in indexes if periods[
            index] < 0.5 * (time_series[-1] - time_series[0])]
        if pmin is not None:
            indexes = [index for index in indexes if periods[index] > pmin]
        if pmax is not None:
            indexes = [index for index in indexes if periods[index] < pmax]

        if running_noise_threshold is not None:
            # prune peaks below running noise if given
            indexes = [index for index in indexes if ft[
                index] > running_noise_threshold[index]]
            signal_to_noise = [
                ft[index] / running_noise_threshold[index] for index in indexes]

            indexes = [index for index, _ in sorted(zip(indexes, signal_to_noise), key=lambda pair: pair[1],
                                                    reverse=True)]  # sort peaks by signal to noise
        else:
            indexes = [index for index, _ in sorted(zip(indexes, [ft[index] for index in indexes]),
                                                    key=lambda pair: pair[1], reverse=True)]  # sort peaks by amplitude

        # trim to n most significant peaks (default None => keep all)
        indexes = indexes[:num_periods]

        if running_noise_threshold is not None:
            signal_to_noise = [
                ft[index] / running_noise_threshold[index] for index in indexes]

        self.ft = ft
        self.period_axis = periods
        self.peak_indexes = indexes
        self.peak_signal_to_noise = signal_to_noise
        self.periods = [periods[index] for index in indexes]
        self.peak_size = [ft[index] for index in indexes]
        return

    def plot_data_autocol_ft(self, use_binned_data=True, interactive=False):
        utils.save_autocorrelation_plots(self.timeseries_binned if use_binned_data else self.timeseries,
                                         self.flux_binned if use_binned_data else self.flux,
                                         self.lag_timeseries, self.correlations,
                                         self.period_axis, self.ft, self.peak_indexes,
                                         os.path.join(
                                             self.filename, 'autocorrelation.pdf'),
                                         running_max_peak=self.running_noise_threshold, interactive=interactive)
        return

    def plot_error_calculations(self, interactive=False):
        utils.save_autocorrelation_plots_multi(self.timeseries_binned, self.gen_data_dict, self.lag_timeseries,
                                               self.gen_autocol_dict, self.period_axis,
                                               self.gen_ft_dict,
                                               os.path.join(self.filename, [
                                                            'errors', 'autocorrelation.pdf']),
                                               interactive=interactive)
        return

    def plot_data(self, use_binned_data=True, interactive=False):
        utils.save_data_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                             self.flux_binned if use_binned_data else self.flux,
                             filename=os.path.join(self.filename, 'data.pdf'), interactive=interactive)
        return

    def plot_autocol(self, interactive=False):
        utils.save_autocorrelation_plot(self.lag_timeseries, self.correlations,
                                        filename=os.path.join(
                                            self.filename, 'autocorrelation_function.pdf'),
                                        interactive=interactive)
        return

    def plot_ft(self, interactive=False):
        utils.save_ft_plt(self.period_axis, self.ft, self.peak_indexes,
                          filename=os.path.join(
                              self.filename, 'fourier_transform.pdf'),
                          running_noise_threshold=self.running_noise_threshold, interactive=interactive)
        return

    def plot_phase_folded_lcs(self, use_binned_data=True, interactive=False):
        for i, period in enumerate(self.periods):
            utils.save_phase_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                                  period, 0,
                                  self.flux_binned if use_binned_data else self.flux,
                                  signal_to_noise=self.peak_signal_to_noise[
                                      i] if self.peak_signal_to_noise else None,
                                  filename=os.path.join(
                                      self.filename, 'phase_fold_{}_days.pdf'.format(period)),
                                  interactive=interactive)
        return

    def plot_phase_folded_lc(self, period, epoch=0, signal_to_noise=None, use_binned_data=True, interactive=False):
        utils.save_phase_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                              period, epoch,
                              self.flux_binned if use_binned_data else self.flux,
                              filename=os.path.join(
                                  self.filename, 'phase_fold_{}_days.pdf'.format(period)),
                              signal_to_noise=signal_to_noise, interactive=interactive)
        return

    def pickle_object(self, pickle_file=None):
        if pickle_file is None:
            pickle_file = os.path.join(self.filename, [self, '.pkl'])
        pickle.dump(self, open(pickle_file, 'w'))
        return

    def object_to_json(self):
        return json.dumps(self, cls=utils.ClassEncoder)

    def log_peaks(self):
        self.safe_log('peak periods: {}'.format(self.periods))
        self.safe_log('peak sizes: {}'.format(self.peak_size))
        self.safe_log('signal to noise {}'.format(self.peak_signal_to_noise))
        return

    def save_and_log_object(self, reduce_data=True):
        if self.logger is None:
            self.create_logger()
        self.log_peaks()
        self.remove_logger()  # to allow JSON serialisation must remove logger

        if reduce_data:  # keep only periods, signal to noise, metadata
            attr = self.__dict__.keys()
            attrs_to_keep = ['periods', 'peak_size', 'filename', 'obj', 'field',
                             'cleaned_refined_periods', 'test', 'ok', 'message',
                             'tbin', 'alpha', 'lag_resolution', 'max_lag', 'min_lag']
            attr_to_remove = [a for a in attr if a not in attrs_to_keep]

            for a in attr_to_remove:
                delattr(self, a)

        jso = self.object_to_json()
        with open(os.path.join(self.filename, 'object.json'), 'w') as json_file:
            json_file.write(jso)

        return

    def clean_periods(self, n_periods=1):
        t = np.array(self.lag_timeseries)
        x = np.array(self.correlations)
        prs = []
        if self.periods:
            p_count = 0
            for p, a in zip(self.periods, self.peak_size):
                if p_count >= n_periods:
                    break
                try:
                    if p < 1.0:
                        # remove periods < 1 day for now
                        continue
                    pr = utils.refine_period(t, x, p)
                    self.safe_log('Altered {} to {}'.format(p, pr))
                    pr = pr.round(decimals=3)
                    if (pr, a) not in prs and pr < 0.5 * (self.timeseries_binned[-1] - self.timeseries_binned[0]):
                        prs.append((pr, a))
                except Exception as e:
                    self.safe_log('Period {} not refined - ignoring'.format(p))
                    continue
                p_count += 1
            max_amp = prs[0][1]
            self.refined_periods = [p for (p, a) in prs if (a > 0.8 * max_amp)]
            self.cleaned_refined_periods = utils.clear_aliases(self.refined_periods, 1.0,
                                                               0.1)  # remove 1 day aliases to 0.1 days
            print_list = np.array2string(
                np.array(self.cleaned_refined_periods))
            self.safe_log('Most likely periods {}'.format(print_list))
            return self.cleaned_refined_periods

    def check_moon_detection(self, moon_epoch, period=None, shape_or_noise=True):
        is_moon = True
        if period is None:
            if len(self.cleaned_refined_periods) > 0:
                period = self.cleaned_refined_periods[0]
            else:
                is_moon = False
#                 self.safe_log('No Periods Found')
                return is_moon
        # is period within range 25 - 30 days?
        if not 25.0 < period < 32.0:
            is_moon = False
            self.safe_log('NOT MOON: Rejected on period range')
            return is_moon
        # is there a noticable change in the LC around 0.5 phase from new moon?
        t = self.timeseries_binned
        f = self.flux_binned
        phase_app, data_app = utils.create_phase(t, period, moon_epoch), f
        binned_phase_app, binned_data_app = utils.bin_phase_curve(
            phase_app, data_app)

        idx_in = np.array([i for i in range(len(phase_app))
                           if 0.4 < phase_app[i] < 0.6])
        idx_out = np.array(
            [i for i in range(len(phase_app)) if i not in idx_in])
        idx_in_bin = [i for i in range(
            len(binned_phase_app)) if 0.4 < binned_phase_app[i] < 0.6]
        idx_out_bin = [i for i in range(
            len(binned_phase_app)) if i not in idx_in_bin]

        med_out, sig_out = utils.medsig(np.array(data_app)[idx_out])
        med_in, sig_in = utils.medsig(np.array(data_app)[idx_in])

        mean_diff_in = np.mean(np.diff(binned_data_app[idx_in_bin]))
        mean_diff_out = np.mean(np.diff(binned_data_app[idx_out_bin]))

#         fig, ax = plt.subplots(figsize=(10,5))
#         ax.scatter(phase_app[idx_in], data_app[idx_in], s=0.1, c='r')
#         ax.scatter(phase_app[idx_out], data_app[idx_out], s=0.1, c='g')
#         ax.axhline(med_in, color='r')
#         ax.axhline(med_out, color='g')
#         ax.axhspan(med_in-sig_in, med_in+sig_in, color='r', alpha=0.2)
#         ax.axhspan(med_out-sig_out, med_out+sig_out, color='g', alpha=0.2)
#         plt.show()
    #     print 'Outside: {}  {}'.format(med_out, sig_out)
    #     print 'Inside: {}  {}'.format(med_in, sig_in)

        if (not sig_in > sig_out):
            self.safe_log('NOT MOON: Rejected on noise threshold')
            noise_crit = False
        else:
            self.safe_log('MOON: Noise criterion suggests moon')
            noise_crit = True
        if (not abs(med_in - med_out) > sig_out):
            self.safe_log('NOT MOON: Rejected on signal shape')
            shape_crit = False
    #         print 'Outside: {}  {}'.format(med_out, sig_out)
    #         print 'Inside: {}  {}'.format(med_in, sig_in)
        else:
            self.safe_log('MOON: Signal shape suggests moon')
            shape_crit = True
            if med_in > med_out:
                self.safe_log('MOON SIGNAL HIGHER')
            else:
                self.safe_log('MOON SIGNAL LOWER')
        if mean_diff_in <= mean_diff_out:
            self.safe_log('NOT MOON: Rejected on flatness criterion')
            diff_crit = False
        else:
            self.safe_log('MOON: Flatness criterion suggests moon')
            diff_crit = True

        is_moon = True if sum(
            1 for ct in [noise_crit, diff_crit, shape_crit] if ct) >= 2 else False
        if is_moon:
            self.safe_log('Moon period removed: {} days'.format(period))
        return is_moon

    def check_fourier_ok(self, nlim=20, peakpct=0.5, npeakstocheck=3, binary_check=0.8):
        is_ok = True
        ft_max = max(self.ft)
        peaks = self.ft[self.peak_indexes] / ft_max
    #     Are peaks sorted? - not sure if needed check
    #     if not np.array_equal(peaks,sorted(peaks, reverse=True)):
    #         is_ok = False
    #         print 'Rejected as peaks not sorted in size => weirdness'
    #         return is_ok
        # hard limit on number of peaks when it will be noise
        if len(peaks) > nlim:
            is_ok = False
            self.safe_log(
                'BAD FFT: Rejected on no of peaks (>{})'.format(nlim))
            return is_ok
        elif len(peaks) == 1:
            self.safe_log('GOOD FFT: Only 1 period found, ok!')
            return is_ok
        elif len(peaks) == 0:
            self.safe_log('BAD FFT: No peaks found')
            is_ok = False
            return is_ok
        # is the max peak > peakpct?
        if max(peaks) < peakpct:
            is_ok = False
            self.safe_log(
                'BAD FFT: Rejected on highest peak not being significant (peakpct)')
            return is_ok
        # is the second peak similar to the first peak?
        if peaks[1] > binary_check:
            self.safe_log('FFT CHECK: Possible binary detected')
        # is the highest peak significantly above the mean difference between
        # peaks?
        meandiff = np.mean(abs(np.diff(peaks)))
        if abs(peaks[0] - peaks[1]) < meandiff:
            is_ok = False
            self.safe_log(
                'BAD FFT: Rejected on highest peak not being significant (diff)')
            return is_ok
        return is_ok

    def check_signal_significance(self, p=None, epoch=0, signal_threshold=2.5,
                                  phase_threshold=1.4, phase_period_threshold=2.5,
                                  return_ratios=False):
        if p is None:
            p = self.cleaned_refined_periods[0]
        tref = (Time(Time(np.median(self.timeseries_binned) + utils.NGTS_EPOCH, format='jd')
                     .to_datetime().replace(hour=12, minute=0, second=0)).jd - utils.NGTS_EPOCH)
        phase_app, data_app = utils.append_to_phase(utils.create_phase(
            self.timeseries_binned, p, epoch), self.flux_binned)
        pref = np.mod(tref - epoch, p) / p
        spread1 = utils.split_and_compute_percentile_per_specified_time_period(self.timeseries_binned,
                                                                               self.flux_binned, tref, 1, nsigma=3)  # nightly spread
        spread2 = utils.split_and_compute_percentile_per_specified_time_period(self.timeseries_binned,
                                                                               self.flux_binned, tref, p)  # spread in 'signal
        signal_spread_ratio = spread2 / spread1
        spread1_phase = utils.split_and_compute_percentile_per_specified_time_period(
            phase_app, data_app, 0, 0.05, nsigma=3)
        spread2_phase = utils.split_and_compute_percentile_per_specified_time_period(
            phase_app, data_app, 0, 1)

        phase_spread_ratio = spread2_phase / spread1_phase

        if p >= 15.:
            nperiods = 1
        elif 5. < p < 15.:
            nperiods = 2
        elif 1. < p < 5.:
            nperiods = 3
        else:
            nperiods = 1

        phases, datas, timeseriess = utils.split_phase(
            phase_app, data_app, self.timeseries_binned, nperiods=nperiods)

        spread_ratios = []
        for pp, fp in zip(phases, datas):
            try:
                spread1_phase = utils.split_and_compute_percentile_per_specified_time_period(
                    pp, fp, 0, 0.05, nsigma=3)
                spread2_phase = utils.split_and_compute_percentile_per_specified_time_period(
                    pp, fp, 0, 1)
            except ValueError:
                pp_ratio = np.nan
            else:
                pp_ratio = spread2_phase / spread1_phase
            spread_ratios.append(
                pp_ratio) if pp_ratio != np.inf else spread_ratios.append(np.nan)
#         print spread_ratios
#         print np.nanpercentile(spread_ratios, [10., 50., 90.])
        phased_signal_spread_ratio = np.nanmean(spread_ratios)

        self.safe_log('Signal spread {} times larger than daily spread'.format(
            signal_spread_ratio))
        self.safe_log('Phased signal spread {} times larger than 0.05 phase spread'.format(
            phase_spread_ratio))
        self.safe_log('Phased signal spread per period {} times larger than 0.05 phase spread'.format(
            phased_signal_spread_ratio))

        if (signal_spread_ratio > signal_threshold
                and phase_spread_ratio > phase_threshold
                and phased_signal_spread_ratio > phase_period_threshold):
            return True  # good
            self.safe_log('Signal Threshold Exceeded - Period OK')
        else:
            return False  # bad
            self.safe_log('Signal Below Threshold - Period Rejected')

    def remove_moon_and_check_ok(self, new_moon_epoch, period=None):
        is_ok = False
        if period is None:
            period = self.cleaned_refined_periods[0]
        t = np.array(self.timeseries_binned)
        f = np.array(self.flux_binned)
        ferr = np.array(self.flux_binned_err)
        t2r, f2r, ferr2r, sgf = utils.remove_moon_signal(
            t=t, f=f, ferr=ferr, P=period, epoch=new_moon_epoch)

        self.timeseries_binned_old = self.timeseries_binned
        self.flux_binned_old = self.flux_binned
        self.flux_binned_err_old = self.flux_binned_err

        self.timeseries_binned = t2r
        self.flux_binned = f2r
        self.flux_binned_err = ferr2r

        self.correlations = None
        self.calculate_periods_from_autocorrelation(
            calculate_noise_threshold=False)

        self.clean_periods()
        new_period = self.cleaned_refined_periods[0]

        fourier_ok = self.check_fourier_ok(nlim=5)
        if fourier_ok:
            shape_or_noise = False if abs(new_period - period) > 1.0 else True
            is_moon = self.check_moon_detection(new_moon_epoch)
            if is_moon:
                self.safe_log('Moon detected again afer removal')
            else:
                self.safe_log('New best period detected after moon removal')
                is_ok = True
        else:
            is_ok = False
            self.safe_log('Removing moon left noise')
        return is_ok
