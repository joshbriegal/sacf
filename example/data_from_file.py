from GACF import find_correlation_from_file
import matplotlib.pyplot as plt

if __name__ == '__main__':
    lag_timeseries, correlations, corr = find_correlation_from_file('test_data.dat')
