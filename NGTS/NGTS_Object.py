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


def return_object_from_json_string(json_str):
    if isinstance(json_str, str):
        dic = json.loads(json_str, cls=utils.NGTSObjectDecoder)
    elif isinstance(json_str, dict):
        dic = json_str
    else:
        raise TypeError("Input must be string or dict")
    tobj = NGTSObject(1, 1, 1)  # temp
    tobj.__dict__ = dic
    try:
        tobj.create_logger()  # JSON has no logger info
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
            self.filename = os.path.join(field_filename, "{}_VERSION_{}".format(obj, test))

        # NGTSIO inputs
        self.obj = obj
        self.field = field
        self.test = test
        self.lc_data2get = lc_data2get if lc_data2get is not None else ['HJD', 'SYSREM_FLUX3', 'FLAGS']
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

        self.ok = True

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

    def create_logger(self, logger=None):
        if logger is None:
            fh = logging.FileHandler(os.path.join(self.filename, 'peaks.log'), 'w')
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

    def get_binned_data(self, delete_unbinned=True):
        self.get_data()
        self.bin_data(delete_unbinned=delete_unbinned)
        return

    def get_data_from_dict(self, dic, nsig2keep=None,
                           do_relflux=True):  # if getting data from NGTSio using multiple lcs
        if not isinstance(dic["OBJ_ID"], list):
            self.flux, self.timeseries, self.median_flux, self.flux_std_dev, self.num_observations, self.ok = \
                utils.clean_ngts_data(dic, nsig2keep=nsig2keep, do_relflux=do_relflux)
            return
        else:
            idx, = np.where(dic["OBJ_ID"] == self)
            if idx.size == 0:
                self.ok = False
                return
            if len(idx) > 1:
                raise ValueError("more than one object returned with this ID - check inputs")
            new_dic = {}
            for key in dic.keys():
                try:
                    new_dic[key] = dic[key][idx]
                    if len(new_dic[key]) == 1:
                        new_dic[key] = new_dic[key][0]
                except TypeError:
                    new_dic[key] = dic[key]
                    self.flux, self.timeseries, self.median_flux, self.flux_std_dev, self.num_observations, self.ok = \
                        utils.clean_ngts_data(new_dic, nsig2keep=nsig2keep, do_relflux=do_relflux)
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
        self.median_flux_binned, self.flux_binned_std_dev = utils.medsig(self.flux_binned)
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

        generated_data = utils.generate_random_lightcurves(self.timeseries_binned, self.flux_binned_err, no_samples)

        self.running_noise_threshold, self.gen_data_dict, self.gen_autocol_dict, self.gen_ft_dict = \
            utils.calculate_noise_level(time_series=self.timeseries_binned, generated_data=generated_data,
                                        sigma_level_calc=sigma_level_calc, sigma_level_out=sigma_level,
                                        min_lag=self.min_lag, max_lag=self.max_lag,
                                        lag_resolution=self.lag_resolution, alpha=self.alpha,
                                        logger=None)

        return

    def calculate_autocorrelation(self, use_binned_data=True):
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)
        if self.logger is None:
            self.create_logger()
        if use_binned_data and (self.timeseries_binned is None or self.flux_binned is None):
            if self.timeseries is not None and self.flux is not None:
                self.bin_data()
            else:
                self.get_binned_data()
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

        self.logger.info(
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
                self.calculate_noise_threshold(**kwargs)
            running_noise_threshold = self.running_noise_threshold

        if self.correlations is None or self.lag_timeseries is None:
            self.calculate_autocorrelation()

        time_series = self.timeseries_binned if use_binned_data else self.timeseries
        signal_to_noise = None

        ft, periods, indexes = utils.fourier_transform_and_peaks(self.correlations, self.lag_timeseries, n=128)

        # pruning: user input pmin and pmax, no periods greater than half the length of the time series.
        indexes = [index for index in indexes if periods[index] < 0.5 * (time_series[-1] - time_series[0])]
        if pmin is not None:
            indexes = [index for index in indexes if periods[index] > pmin]
        if pmax is not None:
            indexes = [index for index in indexes if periods[index] < pmax]

        if running_noise_threshold is not None:
            # prune peaks below running noise if given
            indexes = [index for index in indexes if ft[index] > running_noise_threshold[index]]
            signal_to_noise = [ft[index] / running_noise_threshold[index] for index in indexes]

            indexes = [index for index, _ in sorted(zip(indexes, signal_to_noise), key=lambda pair: pair[1],
                                                    reverse=True)]  # sort peaks by signal to noise
        else:
            indexes = [index for index, _ in sorted(zip(indexes, [ft[index] for index in indexes]),
                                                    key=lambda pair: pair[1], reverse=True)]  # sort peaks by amplitude

        indexes = indexes[:num_periods]  # trim to n most significant peaks (default None => keep all)

        if running_noise_threshold is not None:
            signal_to_noise = [ft[index] / running_noise_threshold[index] for index in indexes]

        self.ft = ft
        self.period_axis = periods
        self.peak_indexes = indexes
        self.peak_signal_to_noise = signal_to_noise
        self.periods = [periods[index] for index in indexes]
        return

    def plot_data_autocol_ft(self, use_binned_data=True, interactive=False):
        utils.save_autocorrelation_plots(self.timeseries_binned if use_binned_data else self.timeseries,
                                         self.flux_binned if use_binned_data else self.flux,
                                         self.lag_timeseries, self.correlations,
                                         self.period_axis, self.ft, self.peak_indexes,
                                         os.path.join(self.filename, 'autocorrelation.pdf'),
                                         running_max_peak=self.running_noise_threshold, interactive=interactive)
        return

    def plot_error_calculations(self, interactive=False):
        utils.save_autocorrelation_plots_multi(self.timeseries_binned, self.gen_data_dict, self.lag_timeseries,
                                               self.gen_autocol_dict, self.period_axis,
                                               self.gen_ft_dict,
                                               os.path.join(self.filename, ['errors', 'autocorrelation.pdf']),
                                               interactive=interactive)
        return

    def plot_data(self, use_binned_data=True, interactive=False):
        utils.save_data_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                             self.flux_binned if use_binned_data else self.flux,
                             filename=os.path.join(self.filename, 'data.pdf'), interactive=interactive)
        return

    def plot_autocol(self, interactive=False):
        utils.save_autocorrelation_plot(self.lag_timeseries, self.correlations,
                                        filename=os.path.join(self.filename, 'autocorrelation_function.pdf'),
                                        interactive=interactive)
        return

    def plot_ft(self, interactive=False):
        utils.save_ft_plt(self.period_axis, self.ft, self.peak_indexes,
                          filename=os.path.join(self.filename, 'fourier_transform.pdf'),
                          running_noise_threshold=self.running_noise_threshold, interactive=interactive)
        return

    def plot_phase_folded_lcs(self, use_binned_data=True, interactive=False):
        for i, period in enumerate(self.periods):
            utils.save_phase_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                                  period, 0,
                                  self.flux_binned if use_binned_data else self.flux,
                                  signal_to_noise=self.peak_signal_to_noise[i] if self.peak_signal_to_noise else None,
                                  filename=os.path.join(self.filename, 'phase_fold_{}_days.pdf'.format(period)),
                                  interactive=interactive)
        return

    def plot_phase_folded_lc(self, period, epoch=0, signal_to_noise=None, use_binned_data=True, interactive=False):
        utils.save_phase_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                              period, epoch,
                              self.flux_binned if use_binned_data else self.flux,
                              filename=os.path.join(self.filename, 'phase_fold_{}_days.pdf'.format(period)),
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
        self.logger.info('peak periods: {}'.format(self.periods))
        self.logger.info('signal to noise {}'.format(self.peak_signal_to_noise))
        return

    def save_and_log_object(self, clear_unneeded_data=False):
        if self.logger is None:
            self.create_logger()
        self.log_peaks()
        self.remove_logger()  # to allow JSON serialisation must remove logger
        jso = self.object_to_json()
        json_file = open(os.path.join(self.filename, 'object.json'), 'w')
        json_file.write(jso)
        json_file.close()
        if clear_unneeded_data:  # keep only periods, signal to noise, metadata
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
            self.ft, self.period_axis = None, None
            self.peak_indexes, self.peak_percentages = None, None
        return
