from datastructure import DataStructure
from correlator import CorrelationIterator, Correlator

SELECTION_FUNCTIONS = {
    'fast': Correlator.fastSelectionFunctionIdx,
    'natural': Correlator.naturalSelectionFunctionIdx
}

WEIGHT_FUNCTIONS = {
    'gaussian': Correlator.getGaussianWeights,
    'fractional': Correlator.getFractionWeights
}


def set_up_correlation(corr, max_lag=None, lag_resolution=None, alpha=None):
    """ No return type. Applies non-default values to correlator """
    if max_lag is not None:
        corr.max_lag = max_lag
    if lag_resolution is not None:
        corr.lag_resolution = lag_resolution
    if alpha is not None:
        corr.alpha = alpha


def find_correlation(corr, selection_function='natural', weight_function='gaussian'):
    k = 0
    i = 1
    number_of_steps = corr.max_lag / corr.lag_resolution
    while k < corr.max_lag:
        col_it = CorrelationIterator(k)
        SELECTION_FUNCTIONS[selection_function](corr, col_it)
        corr.deltaT(col_it)
        WEIGHT_FUNCTIONS[weight_function](corr, col_it)
        corr.findCorrelation(col_it)
        corr.addCorrelationData(col_it)
        k += corr.lag_resolution
        i += 1


def find_correlation_from_file(filename, max_lag=None, lag_resolution=None, selection_function='natural',
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

    set_up_correlation(corr, max_lag, lag_resolution, alpha)
    find_correlation(corr, selection_function, weight_function)
    return {'lag_timeseries': corr.lag_timeseries(), 'correlations': corr.correlations()}


def find_correlation_from_lists(values, timeseries, errors=None, max_lag=None, lag_resolution=None,
                                selection_function='natural', weight_function='gaussian', alpha=None):
    """
    :param values: list of X values
    :param timeseries: list of time values
    :param errors: (optional) list of errors on X values
    :param max_lag: max lag in days
    :param lag_resolution: lag resolution in days
    :param selection_function: 'fast' or 'natural' - see paper for more details
    :param weight_function: 'gaussian' or 'fractional' see paper for more details
    :param alpha: weight function characteristic length scale, default is t.median_time
    :return: { 'lag_timeseries':[], 'correlations':[] }
    """
    if errors is None:
        ds = DataStructure(values, timeseries)
    else:
        ds = DataStructure(values, timeseries, errors)
    corr = Correlator(ds)

    set_up_correlation(corr, max_lag, lag_resolution, alpha)
    find_correlation(corr, selection_function, weight_function)
    return {'lag_timeseries': corr.lag_timeseries(), 'correlations': corr.correlations()}