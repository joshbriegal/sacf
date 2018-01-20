from GACF.datastructure import DataStructure
from GACF.correlator import CorrelationIterator, Correlator
import matplotlib.pyplot as plt

filename = 'NG0522-2518_031001_LC_tbin=10min.dat'
ds = DataStructure(filename)
print 'loaded data {} to DataStructure object ds'.format(filename)
c = Correlator(ds)
print 'setup correlation for file {} with properties: \n alpha = {} \n max lag = {} \n lag resolution = {}'.format(filename, c.alpha, c.max_lag, c.lag_resolution)

def find_correlation(correlator):
    k = 0
    i = 1
    number_of_steps = correlator.max_lag / correlator.lag_resolution
    while k<correlator.max_lag:
        col_it = CorrelationIterator(k)
        correlator.naturalSelectionFunctionIdx(col_it)
        correlator.deltaT(col_it)
        correlator.getGaussianWeights(col_it)
        correlator.findCorrelation(col_it)
        correlator.addCorrelationData(col_it)
        print 'Correlation found for timestep {} of {}'.format(i, number_of_steps)
        k += correlator.lag_resolution
        i += 1

find_correlation(c)
plt.plot(c.lag_timeseries(), c.correlations())
ax = plt.gca()
ax.set_title('Autocorrelations for file {}'.format(filename))
ax.set_xlabel('Lag / Days')
ax.set_ylabel('Autocorrelation')
plt.show()

