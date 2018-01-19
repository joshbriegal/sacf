# Generalised Autocorrelation Function 
(credit Larz Kreutzer, c++ implementation by Josh Briegal jtb34@cam.ac.uk)

## Installation
Only requirement for installation is CMAKE (https://cmake.org).
From above top level run 'pip install ./GACF'

in python:

from GACF import *

correlation_dictionary = find_correlation_from_file('filepath')
OR
correlation_dictionary = find_correlation_from_lists(values, timeseries, errors=None)

with options:

max_lag=None, lag_resolution=None, selection_function='natural', weight_function='gaussian', alpha=None