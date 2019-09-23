from GACF import find_correlation_from_lists, find_correlation_from_lists_cpp
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


def get_aliases(freq, sampling_freq, lower_limit=None, upper_limit=None, num_aliases=3):
    aliases = []
    if lower_limit is None:
        lower_limit = freq - num_aliases * sampling_freq
    #     if lower_limit < 0:
    #         lower_limit = 0
    #         print 'lower limit 0'
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


def clear_aliases(prs, p_sampling, tolerance=1e-3):
    # input sorted list of periods (assume first peroid is not an alias?)
    not_alias = np.ones(len(prs), dtype=bool)
    for i, p in enumerate(prs):
        if not_alias[i]:
            p_aliases = np.divide(1.0, get_aliases(1.0 / p, 1.0 / p_sampling, num_aliases=5))
            aliases = []
            for j, pp in enumerate(prs[:i] + prs[(i + 1):]):
                idx = j if j < i else j + 1
                tru = np.isclose(p_aliases, pp, atol=tolerance)
                not_alias[idx] = ~np.any(tru) * not_alias[idx]
    return [p for b, p in zip(not_alias, prs) if b]


def fourier_transform_and_peaks(correlations, lag_timeseries, len_ft=None, n=32, power_of_2=True, pad_both_sides=True):
    if len_ft is None:
        len_ft = len(lag_timeseries) * n

    if power_of_2:
        # add more zeros to make the length of the array a power of 2, for computational speed up
        len_ft = int(2.0 ** np.ceil(np.log2(len_ft)))

    # print 'number of samples', len_ft, '2 ** ', np.log2(len_ft)

    if pad_both_sides:
        num_zeros = len_ft - len(lag_timeseries)
        zeros = np.zeros(num_zeros / 2)
        correlations = np.append(zeros, np.append(correlations, zeros))

    complex_ft = fft.rfft(correlations, n=len_ft)
    freqs = fft.rfftfreq(len_ft, lag_timeseries[1] - lag_timeseries[0])

    with np.errstate(divide='ignore', invalid='ignore'):
        # will warn divide by zero otherwise
        periods = 1 / freqs
    ft = np.abs(complex_ft)
    
    # return only part of FFT up to length of timeseries
    idx_ok = periods < max(lag_timeseries)
    periods = periods[idx_ok]
    ft = ft[idx_ok]

    # Find peaks of FFT

    indexes = peakutils.indexes(ft, thres=0.3,  # Fraction of largest peak
                                min_dist=3  # Number of data points between
                               )

    return ft, periods, indexes


def refine_period(t, x, p, n=5):
    idxs = np.where(np.abs(t) < n * p)[0]
    x2 = np.array(x)[idxs]
    t2 = np.array(t)[idxs]

    hm = np.hamming(len(x2))
    x2 = hm * x2

    ft2, p2, i2 = fourier_transform_and_peaks(x2, t2)
    ps2 = np.array([pair[0] for pair in sorted([q for q in zip(p2[i2], ft2[i2])], key=lambda x: x[1], reverse=True)])
    pc = ps2[np.argmin(np.abs(ps2 - p))]

    #     fig, ax = plt.subplots(figsize=(15, 3))
    #     ax.plot(p2, ft2, marker='o', ms=1, lw=1)
    #     ax.set_xscale('log')
    #     # ax.set_yscale('log')
    #     ax.axvline(x=pc, lw=1, c='r')
    #     plt.show()
    #     print 'Out:', ps2

    return pc


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


def calculate_noise_level(time_series, generated_data, sigma_level_calc=3, sigma_level_out=5,
                          min_lag=None, max_lag=None, lag_resolution=None, alpha=None, logger=None):
    sigma_multiplication_factor = float(sigma_level_out) / float(sigma_level_calc)
    no_samples = sigma_level_to_samples(sigma_level=sigma_level_calc)

    lag_timeseries, generated_correlations, _ = find_correlation_from_lists_cpp(time_series, generated_data,
                                                                                min_lag=min_lag,
                                                                                max_lag=max_lag,
                                                                                lag_resolution=lag_resolution,
                                                                                alpha=alpha)

    #     lag_timeseries = generated_correlations['lag_timeseries']
    #     generated_correlations = np.array(generated_correlations['correlations'])

    generated_periods, generated_ft = None, None
    generated_correlations = np.array([np.array(c) for c in generated_correlations])  # convert to numpy array

    for i, correlations in enumerate(generated_correlations):

        ft, periods, _ = fourier_transform_and_peaks(correlations, lag_timeseries, n=128)

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
