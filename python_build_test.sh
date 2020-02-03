#!/usr/bin/env bash
# make virtualenv
mkdir ./venv
virtualenv --python=python3.7 venv
source venv/bin/activate

# install gacf
pip install -r requirements.txt -r development_requirements.txt
cd ..
pip install ./gacf -v

# test
pytest gacf/gacf/test

# remove venv
rm -r gacf/venv