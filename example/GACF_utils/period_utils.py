from GACF import find_correlation_from_lists
import numpy as np
import numpy.fft as fft
import peakutils
from tqdm import tqdm


def find_peaks_around_period(time_series, x_values, period,
                             num_peaks=3, fraction_around_period=0.5):
    confirm_periods = []
    p_diff = period * fraction_around_period
    for i in range(1, num_peaks + 1):
        centre_time = i * period
        start_time = np.argmin(abs(time_series - (centre_time - p_diff)))
        end_time = np.argmin(abs(time_series - (centre_time + p_diff)))
        if end_time == len(time_series) - 1:
            break
        if start_time != end_time:
            max_period_idx = np.argmax(x_values[start_time:end_time]) + start_time
        else:
            max_period_idx = start_time
        confirm_periods.append(time_series[max_period_idx] / i)

    peak = np.mean(confirm_periods)
    return peak, confirm_periods


def confirm_periods(periods, indexes, correlation_timeseries, correlation_data,
                    num_peaks=3, fraction_around_period=0.5):
    peaks_x = []
    for i, period in enumerate([periods[index] for index in indexes]):
        peak, confirm_periods = find_peaks_around_period(
            correlation_timeseries, correlation_data, period, num_peaks=num_peaks,
            fraction_around_period=fraction_around_period)
        peaks_x.append(peak)

    return peaks_x


def fourier_transform_and_peaks(correlations, lag_timeseries, len_ft=None):
    if len_ft is None:
        len_ft = len(lag_timeseries) * 2

    complex_ft = fft.rfft(correlations, n=len_ft)
    freqs = fft.rfftfreq(len_ft, lag_timeseries[1] - lag_timeseries[0])

    periods = 1 / freqs
    ft = np.abs(complex_ft)

    # Find peaks of FFT

    indexes = peakutils.indexes(ft, thres=0.1,  # Fraction of largest peak
                                min_dist=3  # Number of data points between
                                )

    return ft, periods, indexes


def generate_random_lightcurves(time_series, errors, no_samples=1000):
    generated_data = np.empty(shape=(no_samples, len(time_series)))

    for i in range(no_samples):
        generated_data[i] = np.array([np.random.normal(scale=rms) for rms in errors], dtype=float)

    return generated_data


def sigma_level_to_samples(sigma_level=None, samples=None):
    sigma_to_samples = {
        1: 68,
        2: 96,
        3: 998,
        4: 9999,
        5: 999999
    }

    if sigma_level is not None and samples is None:

        if sigma_level not in sigma_to_samples.keys():
            sigma_values = np.array(sigma_to_samples.keys())
            sigma_level = sigma_values[(np.abs(sigma_values - samples)).argmin()]

        return sigma_to_samples[sigma_level]

    elif samples is not None and sigma_level is None:
        samples_to_sigma = {value: key for key, value in sigma_to_samples.items()}

        if samples not in samples_to_sigma.keys():
            sample_values = np.array(samples_to_sigma.keys())
            samples = sample_values[(np.abs(sample_values - samples)).argmin()]

        return samples_to_sigma[samples]
    else:
        return None


def get_noise_level(time_series, generated_data, sigma_level_calc=3, sigma_level_out=5, logger=None):
    sigma_multiplication_factor = float(sigma_level_out) / float(sigma_level_calc)
    no_samples = sigma_level_to_samples(sigma_level=sigma_level_calc)

    generated_correlations, _ = find_correlation_from_lists(time_series, generated_data)
    lag_timeseries = generated_correlations['lag_timeseries']
    generated_correlations = np.array(generated_correlations['correlations'])

    generated_periods, generated_data, generated_ft = None, None, None

    for i, correlations in enumerate(tqdm(generated_correlations,
                                          total=no_samples, desc='Calculating Noise Fourier Transforms')):

        ft, periods, _ = fourier_transform_and_peaks(correlations, lag_timeseries)

        if i == 0:
            generated_periods = periods
            generated_ft = np.empty(shape=(no_samples, len(ft)))

        generated_ft[i] = ft

    gen_data_dict = {}
    gen_autocol_dict = {}
    gen_ft_dict = {}

    data_transpose = np.array([generated_data[:, i] for i, _ in enumerate(time_series)])
    gen_data_dict['max'] = [max(data) for data in data_transpose]
    gen_data_dict['min'] = [min(data) for data in data_transpose]
    gen_data_dict['median'] = [np.median(data) for data in data_transpose]

    autocol_transpose = np.array([generated_correlations[:, i] for i, _ in enumerate(lag_timeseries)])
    gen_autocol_dict['max'] = [max(autocol) for autocol in autocol_transpose]
    gen_autocol_dict['min'] = [min(autocol) for autocol in autocol_transpose]
    gen_autocol_dict['median'] = [np.median(autocol) for autocol in autocol_transpose]
    gen_autocol_dict['lag_timeseries'] = lag_timeseries

    gen_ft_transpose = np.array([generated_ft[:, i] for i, _ in enumerate(generated_periods)])
    gen_ft_dict['max'] = [max(gen_ft) for gen_ft in gen_ft_transpose]
    gen_ft_dict['min'] = [min(gen_ft) for gen_ft in gen_ft_transpose]
    gen_ft_dict['median'] = [np.median(gen_ft) for gen_ft in gen_ft_transpose]
    gen_ft_dict['periods'] = generated_periods

    running_noise_threshold = np.multiply(gen_ft_dict['max'], sigma_multiplication_factor)

    if logger is not None:
        logger.info('Running: {}'.format(running_noise_threshold.tolist()))

    return running_noise_threshold, gen_data_dict, gen_autocol_dict, gen_ft_dict
