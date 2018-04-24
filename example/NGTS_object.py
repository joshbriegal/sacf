"""
PyObject per NGTS object, containing light curve and associated analysis
"""
import numpy as np
from GACF import find_correlation_from_lists
import GACF_utils as utils
import os


class NGTSObject(object):

    def __init__(self, obj, field, test, lc_data2get=None, tbin=20, nsig2keep=None, do_relflux=True,
                 max_lag=None, lag_resolution=None, selection_function='natural', weight_function='gaussian',
                 alpha=None, field_filename=None):

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

        self.ok = True

    def __str__(self):
        return "NGTS Object " + self.field + "_" + str(self.obj) + " ({})".format(self.test)

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

    def get_binned_data(self, delete_unbinned=True):
        self.get_data()
        self.bin_data(delete_unbinned=delete_unbinned)
        return

    def get_data_from_dict(self, dic, nsig2keep=None,
                           do_relflux=True):  # if getting data from NGTSio using multiple lcs
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
            utils.rebin_err(self.timeseries, self.flux,
                            dt=self.tbin * utils.TIME_CONVERSIONS['m2d'], get_err_on_mean=True)
        self.median_flux_binned, self.flux_binned_std_dev = utils.medsig(self.flux_binned)
        self.num_observations_binned = len(self.timeseries_binned)
        if delete_unbinned:  # should save space as we don't need unbinned data after binning
            self.flux = None
            self.timeseries = None
            self.median_flux = None
            self.flux_std_dev = None
            self.num_observations = None
        return

    def get_noise_threshold(self, sigma_level=5, sigma_level_calc=3):
        if self.flux_binned_err is None:
            if self.flux is not None and self.timeseries is not None:
                self.bin_data()
            else:
                self.get_binned_data()

        no_samples = utils.sigma_level_to_samples(sigma_level=sigma_level_calc)

        generated_data = utils.generate_random_lightcurves(self.timeseries_binned, self.flux_binned_err, no_samples)

        self.running_noise_threshold, self.gen_data_dict, self.gen_autocol_dict, self.gen_ft_dict = \
            utils.get_noise_level(time_series=self.timeseries_binned, generated_data=generated_data,
                                  sigma_level_calc=sigma_level_calc, sigma_level_out=sigma_level, logger=None)

        return

    def get_autocorrelation(self, use_binned_data=True):
        if use_binned_data and (self.timeseries_binned is None or self.flux_binned is None):
            if self.timeseries is not None and self.flux is not None:
                self.bin_data()
            else:
                self.get_binned_data()
        elif not use_binned_data and (self.flux is None or self.timeseries is None):
            self.get_data()

        if use_binned_data:
            correlations, _ = find_correlation_from_lists(self.timeseries_binned, self.flux_binned,
                                                          max_lag=self.max_lag,
                                                          lag_resolution=self.lag_resolution,
                                                          selection_function=self.selection_function,
                                                          weight_function=self.weight_function, alpha=self.alpha)
        else:
            correlations, _ = find_correlation_from_lists(self.timeseries, self.flux,
                                                          max_lag=self.max_lag,
                                                          lag_resolution=self.lag_resolution,
                                                          selection_function=self.selection_function,
                                                          weight_function=self.weight_function, alpha=self.alpha)
        self.correlations = correlations['correlations']
        self.lag_timeseries = correlations['lag_timeseries']

        return

    def get_periods_from_autocorrelation(self, running_noise_threshold=None, num_periods=5, use_binned_data=True):
        if self.correlations is None or self.lag_timeseries is None:
            self.get_autocorrelation()

        if running_noise_threshold is None:
            running_noise_threshold = self.running_noise_threshold

        time_series = self.timeseries_binned if use_binned_data else self.timeseries
        signal_to_noise = None

        ft, periods, indexes = utils.fourier_transform_and_peaks(self.correlations, self.lag_timeseries)

        indexes = [index for index in indexes if periods[index] > 1 and index != 1]  # prune short periods & last point
        indexes = [index for index in indexes if periods[index] < 0.5 * (time_series[-1] - time_series[0])]

        if running_noise_threshold is not None:
            # prune peaks below running noise if given
            indexes = [index for index in indexes if ft[index] > running_noise_threshold[index]]
            signal_to_noise = [ft[index] / running_noise_threshold[index] for index in indexes]

            indexes = [index for index, _ in sorted(zip(indexes, signal_to_noise), key=lambda pair: pair[1],
                                                    reverse=True)]  # sort peaks by signal to noise
        else:
            indexes = [index for index, _ in sorted(zip(indexes, [ft[index] for index in indexes]),
                                                    key=lambda pair: pair[1], reverse=True)]  # sort peaks by amplitude

        indexes = indexes[:num_periods]  # trim to n most significant peaks

        peak_percentages = [ft[index] / ft[indexes[0]] for index in indexes]
        if running_noise_threshold:
            signal_to_noise = [ft[index] / running_noise_threshold[index] for index in indexes]

        self.ft = ft
        self.period_axis = periods
        self.peak_indexes = indexes
        self.peak_percentages = peak_percentages
        self.peak_signal_to_noise = signal_to_noise
        self.periods = [periods[index] for index in indexes]
        return

    def plot_data_autocol_ft(self, use_binned_data=True):
        utils.save_autocorrelation_plots(self.timeseries_binned if use_binned_data else self.timeseries,
                                         self.flux_binned if use_binned_data else self.flux,
                                         self.lag_timeseries, self.correlations,
                                         self.period_axis, self.ft, self.peak_indexes,
                                         os.path.join(self.filename, 'autocorrelation.png'),
                                         running_max_peak=self.running_noise_threshold)

    def plot_error_calculations(self):
        utils.save_autocorrelation_plots_multi(self.timeseries_binned, self.gen_data_dict, self.lag_timeseries,
                                               self.gen_autocol_dict, self.period_axis,
                                               self.gen_ft_dict,
                                               os.path.join(self.filename, ['errors', 'autocorrelation.png']))

    def plot_data(self, use_binned_data=True):
        utils.save_data_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                             self.flux_binned if use_binned_data else self.flux,
                             filename=os.path.join(self.filename, 'data.png'))

    def plot_autocol(self):
        utils.save_autocorrelation_plot(self.lag_timeseries, self.correlations,
                                        filename=os.path.join(self.filename, 'autocorrelation_function.png'))

    def plot_ft(self):
        utils.save_ft_plt(self.period_axis, self.ft, self.peak_indexes,
                          filename=os.path.join(self.filename, 'fourier_transform.png'),
                          running_noise_threshold=self.running_noise_threshold)

    def plot_phase_folded_lcs(self, use_binned_data=True):
        for i, period in self.periods:
            utils.save_phase_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                                  period, 0,
                                  self.flux_binned if use_binned_data else self.flux,
                                  peak_percentage=self.peak_percentages[i],
                                  filename=os.path.join(self.filename, 'phase_fold_{}_days'.format(period)))

    def plot_phase_folded_lc(self, period, epoch=0, peak_percentage=None, use_binned_data=True):
        utils.save_phase_plot(self.timeseries_binned if use_binned_data else self.timeseries,
                              period, epoch,
                              self.flux_binned if use_binned_data else self.flux,
                              filename=os.path.join(self.filename, 'phase_fold_{}_days'.format(period)),
                              peak_percentage=peak_percentage)


if __name__ == '__main__':
    from time import time

    obj = NGTSObject(1, 1, 1)
    obj.timeseries = np.load('JamesJackmanWork/NG2331-3922_009138_hjd.npy')
    obj.flux = np.load('JamesJackmanWork/NG2331-3922_009138_flux.npy')
    # obj.flux_binned_err = np.load('JamesJackmanWork/NG2331-3922_009138_flux_error.npy')
    # obj.plot_data(use_binned_data=False)
    print 'Binning Data'
    bin_start = time()
    obj.bin_data()
    bin_end = time()
    print "Took {} seconds to bin".format(bin_end - bin_start)
    print 'Finding Noise threshold'
    obj.get_noise_threshold()
    print 'Getting Periods'
    obj.get_periods_from_autocorrelation()
