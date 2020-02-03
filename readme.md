![Python Test Status](https://github.com/joshbriegal/gacf/workflows/Python%20tests/badge.svg)
  
# Generalised Autocorrelation Function 
(credit Larz Kreutzer, c++ implementation by Josh Briegal jtb34@cam.ac.uk)

## Installation
Requirements:
   * CMAKE (https://cmake.org) > 3.8.
   * C++14

From above top level directory run
```
pip install ./gacf
```
in python:

```python
from gacf import *

lag_timeseries, correlations, corr_object = find_correlation_from_file('filepath')
```
OR
```python
lag_timeseries, correlations, corr_object = find_correlation_from_lists(timeseries, values, errors=None)
```
with options:
```python
max_lag=None, lag_resolution=None, selection_function='natural', weight_function='gaussian', alpha=None
```

new c++ implementation, should be faster:
```python
from gacf import *

lag_timeseries, correlations, corr_object = find_correlation_from_lists_cpp(timeseries, values, errors=None)
```

### Examples

random_sine_wave.py creates a randomly sampled sine wave and finds the autocorrelation
using the created functions when importing G-ACF.

data_from_file.py demonstrates loading the data from 'test_data.dat' and calculating the G-ACF.

extract_periods.py creates a sine wave of fixed period and demonstrates two methods for period
extraction:
   * Fast Fourier Transform (FFT). Take FFT of G-ACF and select largest peak
   * Select largest peak in G-ACF (or average across multiple peaks)
