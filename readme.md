![Python Test Status](https://github.com/joshbriegal/sacf/workflows/Python%20tests/badge.svg)

# Selective Estimator for the Autocorrelation Function

S-ACF: A selective estimator for the autocorrelation function of irregularly sampled time series
(credit Lars Kreutzer, c++ implementation by Josh Briegal jtb34@cam.ac.uk)

## Installation

Requirements:

* CMAKE (https://cmake.org) > 3.8.
* C++14

From above top level directory run

```
pip install ./sacf
```

in python:


SACF follows Astropy LombScargle implementation:

```python
from sacf import SACF

lag_timeseries, correlations = SACF(timeseries, values, errors=None).autocorrelation()
```

with options:

```python
sacf.autocorrelation(max_lag=None, lag_resolution=None, selection_function='natural', weight_function='fast', alpha=None)
```

NOTE: If users specify `selection_function="fast"`, `weight_function="fractional_squared"` or `weight_function="gaussian"`, a python implementation of the SACF will be invoked which is considerably slower than the default C++ option.

### Tests

From root directory run:

```python
tox
```
