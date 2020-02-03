from .datastructure import DataStructure
from .correlator import CorrelationIterator, Correlator

SELECTION_FUNCTIONS = {
    'fast': Correlator.fastSelectionFunctionIdx,
    'natural': Correlator.naturalSelectionFunctionIdx
}

WEIGHT_FUNCTIONS = {
    'gaussian': Correlator.getGaussianWeights,
    'fractional': Correlator.getFractionWeights
}

GACF_LOG_MESSAGE = ' ######      ###     ######  ######## \n' \
                   '##    ##    ## ##   ##    ## ##       \n' \
                   '##         ##   ##  ##       ##       \n' \
                   '##   #### ##     ## ##       ######   \n' \
                   '##    ##  ######### ##       ##       \n' \
                   '##    ##  ##     ## ##    ## ##       \n' \
                   ' ######   ##     ##  ######  ##    \n' \
                   '------------------------------\n' \
                   'Number of Data Points: {no_data}\n' \
                   'Number of Lag Timesteps: {no_lag_points}\n' \
                   'Lag Resolution: {lag_resolution}\n' \
                   '------------------------------\n'


def set_up_correlation(corr, min_lag=None, max_lag=None, lag_resolution=None, alpha=None):
    """ No return type. Applies non-default values to correlator """
    if max_lag is not None:
        corr.max_lag = max_lag
    if min_lag is not None:
        corr.min_lag = min_lag
    if lag_resolution is not None:
        corr.lag_resolution = lag_resolution
    if alpha is not None:
        corr.alpha = alpha


def find_correlation(corr, selection_function='natural', weight_function='gaussian'):
    def lag_generator(min_lag, max_lag, lag_resolution):
        k = min_lag
        is_positive = False
        while k <= max_lag:
            if k == 0:
                is_positive = True
            if k > 0 and not is_positive:
                is_positive = True
                yield 0
            yield k
            k += lag_resolution

    num_steps = int(round((corr.max_lag - corr.min_lag) / corr.lag_resolution))

    for i, k in enumerate(lag_generator(corr.min_lag, corr.max_lag, corr.lag_resolution)):
        col_it = CorrelationIterator(k, corr.N_datasets)
        SELECTION_FUNCTIONS[selection_function](corr, col_it)
        corr.deltaT(col_it)
        WEIGHT_FUNCTIONS[weight_function](corr, col_it)
        corr.findCorrelation(col_it)
        corr.addCorrelationData(col_it, i)

    corr.cleanCorrelationData(i + 1)


def find_correlation_from_file(filename, min_lag=None, max_lag=None, lag_resolution=None, selection_function='natural',
                               weight_function='gaussian', alpha=None):
    """
    :param filename: string filename containing data in columns X, t, X_err
    :param max_lag: max lag in days
    :param lag_resolution: lag resolution in days
    :param selection_function: 'fast' or 'natural' - see paper for more details
    :param weight_function: 'gaussian' or 'fractional' see paper for more details
    :param alpha: weight function characteristic length scale, default is t.median_time
    :return: { 'lag_timeseries':[], 'correlations':[] }
    """
    ds = DataStructure(filename)
    corr = Correlator(ds)

    set_up_correlation(corr, min_lag, max_lag, lag_resolution, alpha)
    find_correlation(corr, selection_function, weight_function)
    # return a 3 tuple of lag timeseries, correlations, corr object
    return (corr.lag_timeseries(),
            corr.correlations() if len(corr.correlations()) > 1. else corr.correlations()[0],
            corr)


def find_correlation_from_lists(timeseries, values, errors=None, min_lag=None, max_lag=None, lag_resolution=None,
                                selection_function='natural', weight_function='gaussian', alpha=None):
    """
    :param timeseries: list of time values
    :param values: list of X values
    :param errors: (optional) list of errors on X values
    :param max_lag: max lag in days
    :param lag_resolution: lag resolution in days
    :param selection_function: 'fast' or 'natural' - see paper for more details
    :param weight_function: 'gaussian' or 'fractional' see paper for more details
    :param alpha: weight function characteristic length scale, default is t.median_time
    :return: { 'lag_timeseries':[], 'correlations':[] }
    """
    if errors is None:
        ds = DataStructure(timeseries, values)
    else:
        ds = DataStructure(timeseries, values, errors)
    corr = Correlator(ds)

    set_up_correlation(corr, min_lag, max_lag, lag_resolution, alpha)
    find_correlation(corr, selection_function, weight_function)
    return (corr.lag_timeseries(),
            corr.correlations() if len(corr.correlations()) > 1. else corr.correlations()[0],
            corr)


def find_correlation_from_lists_cpp(timeseries, values, errors=None, min_lag=None, max_lag=None, lag_resolution=None,
                                    alpha=None):
    """
        Using a fixed set up of natural selection function and linear weight function,
        will be faster than the python implementation in general.
       :param timeseries: list of time values
       :param values: list of X values
       :param errors: (optional) list of errors on X values
       :param max_lag: max lag in days
       :param lag_resolution: lag resolution in days
       :param alpha: weight function characteristic length scale, default is t.median_time
       :return: { 'lag_timeseries':[], 'correlations':[] }
       """
    if errors is None:
        ds = DataStructure(timeseries, values)
    else:
        ds = DataStructure(timeseries, values, errors)
    corr = Correlator(ds)
    set_up_correlation(corr, min_lag, max_lag, lag_resolution, alpha)
    corr.calculateStandardCorrelation()
    return (corr.lag_timeseries(),
            corr.correlations() if len(corr.correlations()) > 1. else corr.correlations()[0],
            corr)
