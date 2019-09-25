import sys

# import seaborn as sns
import numpy as np
from scipy import stats, integrate
import os
import matplotlib as mpl
import matplotlib.ticker as tk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import re
from copy import deepcopy
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import pandas as pd
import scipy.interpolate as interpolate
import json
import time
from astropy.time import Time
from tqdm import tqdm
import multiprocessing as mp

plt.rcParams.update({'font.size': 12})
sys.path.append('/home/jtb34/GitHub/GACF/')
#sys.path.append('/home/jtb34/python27')
# print sys.path
from NGTS.NGTS_Field import return_field_from_object_directory, NGTSField
from NGTS.NGTS_Object import NGTSObject
from NGTS.GACF_utils import TIME_CONVERSIONS
import NGTS.GACF_utils as utils

def resample_timeseries(t, f, n=2000):
    temp = np.linspace(0, len(t)-1, len(t)).astype(int)
    np.random.shuffle(temp)
    idxs = sorted(temp[:n])
#     idxs = sorted(np.random.randint(0, len(t), len(t)))
    return np.array(t)[idxs], np.array(f)[idxs]

ROOT_DIR = '/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS'

fieldname = 'NG0612-2518'
obj_id = 40255

# obj = NGTSObject(field=fieldname, obj=obj_id, test='CYCLE1807')
field = NGTSField(fieldname=fieldname, root_file=ROOT_DIR, object_list=[obj_id])
# fits_file, _, _ = find_fits_file(os.path.join(ROOT_DIR, fieldname))
# field.load_from_fits(filename=fits_file, obj_ids=[obj_id], sparse_ids=True, method='fitsio', clean_data=False)
obj = field[obj_id]
obj.ok = True

df = pd.read_csv('5045977_044284.lc.txt', comment='#', sep='\s+', names=['hjd',  'mag',  'mag_err',  'rel_flux',  'rel_flux_err'])

idx_ok = np.isfinite(df['rel_flux'])
obj.flux = df['rel_flux'][idx_ok]
obj.timeseries = df['hjd'][idx_ok] - df['hjd'][0]
obj.bin_data(delete_unbinned=False)

def mp_worker(args):
    torig, forig = args

    n = 32
    periods = []
    for i in tqdm(range(n)):
        obj = NGTSField(fieldname=fieldname, root_file=ROOT_DIR, object_list=[obj_id])[obj_id]
        obj.ok = True
        obj.logger = None
        obj.min_lag = 0
        tt, ff = resample_timeseries(torig, forig)
        obj.num_observations_binned = len(tt)
        obj.timeseries_binned = tt.tolist()
        obj.flux_binned = ff.tolist()
        obj.correlations = None
        obj.lag_timeseries = None
        obj.calculate_periods_from_autocorrelation()
        periods.append(obj.clean_periods()[0])
    
    return periods

if __name__ == '__main__':

    NUM_PROC = 32
    mp_inputs = [(obj.timeseries_binned, obj.flux_binned) for i in range(NUM_PROC)]

    print len(mp_inputs)

    pool = mp.Pool(NUM_PROC)
    total_successes = pool.map(mp_worker, mp_inputs)
    periods = [ent for sublist in total_successes for ent in sublist]

    p_outputs = [p for p in periods if 15<p<20]

    mean = np.mean(p_outputs)
    std = np.std(p_outputs)
    err = std / np.sqrt(len(p_outputs))

    print '{} pm {} days'.format(mean, err)