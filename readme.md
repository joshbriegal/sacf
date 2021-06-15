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

lag_timeseries, correlations, corr_object = find_correlation_from_lists_cpp(timeseries, values, errors=None
```

### Tests

From root directory run:

```python
tox
```
