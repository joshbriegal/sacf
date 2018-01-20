# Generalised Autocorrelation Function 
(credit Larz Kreutzer, c++ implementation by Josh Briegal jtb34@cam.ac.uk)

## Installation
Only requirement for installation is CMAKE (https://cmake.org).
From above top level run 
```
pip install ./GACF
```
in python:

```python
from GACF import *

correlation_dictionary = find_correlation_from_file('filepath')
```
OR
```python
correlation_dictionary = find_correlation_from_lists(values, timeseries, errors=None)
```
with options:
```python
max_lag=None, lag_resolution=None, selection_function='natural', weight_function='gaussian', alpha=None
```
### Examples

function_import_sine_wave_test.py creates a randomly sampled sine wave and finds the autocorrelation
using the created functions when importing GACF

objects_test_from_file.py exposes the underlying c++ object structure to find the correlation of a timeseries from file
