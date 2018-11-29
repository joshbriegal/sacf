#!/usr/bin/env bash
python setup.py test && rm GACF/*.so && rm -r build && rm -r GACF.egg-info