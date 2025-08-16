from .datastructure import DataStructure
from .correlator import CorrelationIterator, Correlator

SELECTION_FUNCTIONS = {
    "fast": Correlator.fastSelectionFunctionIdx,
    "natural": Correlator.naturalSelectionFunctionIdx,
}

WEIGHT_FUNCTIONS = {
    "gaussian": Correlator.getGaussianWeights,
    "fractional": Correlator.getFractionWeights,
    "fractional_squared": Correlator.getFractionSquaredWeights,
}

SACF_LOG_MESSAGE = (
    "███████╗ █████╗  ██████╗███████╗\n"
    "██╔════╝██╔══██╗██╔════╝██╔════╝\n"
    "███████╗███████║██║     █████╗  \n"
    "╚════██║██╔══██║██║     ██╔══╝  \n"
    "███████║██║  ██║╚██████╗██║     \n"
    "╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝     \n"
    "------------------------------\n"
    "Number of Data Points: {no_data}\n"
    "Number of Lag Timesteps: {no_lag_points}\n"
    "Lag Resolution: {lag_resolution}\n"
    "------------------------------\n"
)


class SACF:
    """Compute the Generalised Autocorrelation Function (G-ACF)"""

    def __init__(self, timeseries=None, values=None, errors=None, filename=None):
        if filename:
            self.data = DataStructure(filename)
        else:
            if errors is None:
                self.data = DataStructure(timeseries, values)
            else:
                self.data = DataStructure(timeseries, values, errors)

    @staticmethod
    def set_up_correlation(
        corr, min_lag=None, max_lag=None, lag_resolution=None, alpha=None
    ):
        """No return type. Applies non-default values to correlator"""
        if max_lag is not None:
            corr.max_lag = max_lag
        if min_lag is not None:
            corr.min_lag = min_lag
        if lag_resolution is not None:
            corr.lag_resolution = lag_resolution
        if alpha is not None:
            corr.alpha = alpha

    @staticmethod
    def find_correlation(
        corr, selection_function="natural", weight_function="fractional"
    ):
        """If user specifies a different weight or selection function this method is invoked.
           Will be considerably slower than the C++ implementation.

        Args:
            corr (Correlator): Correlator object to iterate over
            selection_function (str, optional): Selection Function. Defaults to "natural".
            weight_function (str, optional): Weight Function. Defaults to "fractional".
        """

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

        for i, k in enumerate(
            lag_generator(corr.min_lag, corr.max_lag, corr.lag_resolution)
        ):
            col_it = CorrelationIterator(k, corr.N_datasets)
            SELECTION_FUNCTIONS[selection_function](corr, col_it)
            corr.deltaT(col_it)
            WEIGHT_FUNCTIONS[weight_function](corr, col_it)
            corr.findCorrelation(col_it)
            corr.addCorrelationData(col_it, i)

        corr.cleanCorrelationData(i + 1)

    def autocorrelation(
        self,
        min_lag=None,
        max_lag=None,
        lag_resolution=None,
        alpha=None,
        selection_function="natural",
        weight_function="fractional",
        return_correlator=False,
    ):
        """Compute G-ACF

        Using a fixed set up of natural selection function and linear weight function
        will be faster than the python implementation in general.
        It is reccomended to leave selection_function and weight_function as default for speed.

        Args:
            min_lag (float, optional): min lag in units of time. Defaults to None.
            max_lag (float, optional): max lag in units of time. Defaults to None.
            lag_resolution (float, optional): lag resolution in units of time. Defaults to None.
            alpha (float, optional): weight function characteristic length scale, default is t.median_time. Defaults to None.
            selection_function (str, optional): 'fast' or 'natural' - see paper for more details. Defaults to "natural".
            weight_function: (str, optional) 'fractional', 'gaussian' or 'fractional_squared' see paper for more details. Defaults to "fractional".
            return_correlator (bool, optional): return correlator object. Defaults to False.

        Returns:
            Tuple: (lag_timeseries, correlations, [Optional: correlator])
        """
        corr = Correlator(self.data)
        self.set_up_correlation(corr, min_lag, max_lag, lag_resolution, alpha)

        if selection_function == "natural" and weight_function == "fractional":
            # fast c++ method
            corr.calculateStandardCorrelation()
        else:
            # slow python method with user specified function
            self.find_correlation(corr, selection_function, weight_function)

        if return_correlator:
            return (
                corr.lag_timeseries(),
                corr.correlations()
                if len(corr.correlations()) > 1.0
                else corr.correlations()[0],
                corr,
            )
        else:
            return (
                corr.lag_timeseries(),
                corr.correlations()
                if len(corr.correlations()) > 1.0
                else corr.correlations()[0],
            )
