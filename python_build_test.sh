#!/usr/bin/env bash
pip install -r requirements.txt -r development_requirements.txt
cd ..
pip install ./GACF -v
pytest GACF/GACF/tests