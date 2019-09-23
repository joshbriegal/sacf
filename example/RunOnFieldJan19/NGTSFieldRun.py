import sys
import os
import multiprocessing as mp
import numpy as np
import traceback
import time
import re
import json

import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# on server
sys.path.append('/appch/data/jtb34/GitHub/GACF/NGTS/')
from NGTS_Field import NGTSField as NF
from GACF_utils import TIME_CONVERSIONS, refine_period, clear_aliases

# FIELDNAME = 'NG0004-2950'  # 3574 objects, batches of 50
# FIELDNAME = "NG0535-0523"
# FIELDNAME = "NG2058-0248"
FIELDNAME = "NG2346-3633"
# FIELDNAME = "NG2126-1652"
# FIELDNAME = 'NG0442-3345'  # 5059 objects, batches of 100
# FIELDNAME = 'NG2331-3922'  # xxx objects, batches of 25 (already run 1706) TODO
# FIELDNAME = 'NG0504-3633'  # 6137 objects, batches of 25
CYCLE = 'CYCLE1807'
# CYCLE = 'CYCLE1807'
NUM_PROC = min(mp.cpu_count() / 2, 8)


# NUM_PROC = 16


def chunk_list(big_list, small_list_size):
    for i in range(0, len(big_list), small_list_size):
        yield big_list[i:i + small_list_size]


def find_fits_file(field, cycle, directory=None):
    if directory is None:
        directory = os.getcwd()
    string_pattern = r'^{}_\d+_[0-9\-/]+_{}.fits$'.format(field, cycle)
    # print string_pattern
    pattern = re.compile(string_pattern)
    for file in os.listdir(directory):
        # print file
        if re.search(pattern, file) is not None:
            return file
    raise IOError('File not found in directory {}'.format(directory))


def wrapped_worker(args):
    try:
        return worker(args)
    except:
        traceback.print_exc()
        print 'Subprocess failed, exiting'


def worker(args):
    object_list, fits_file, root_file = args
    start = time.time()
    print 'Worker started at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start)))
    field = NF(fieldname=FIELDNAME, test=CYCLE, object_list=object_list, root_file=root_file)
    pdic = {}
    print field, 'Num objects: {}'.format(field.num_objects)
    for obj in field:
        obj.tbin = 30
        obj.lag_resolution = 30 * TIME_CONVERSIONS['m2d']
        try:
            obj.create_logger()
        except IOError:
            pass
    try:
        field.get_object_lc_bulk()
        # print 'Binned', field.object_list
        field.calculate_periods_from_autocorrelation(calculate_noise=False)
        # print 'Correlations calculated', field.object_list

    except Exception as e:
        traceback.print_exc()
        print 'Obj IDs not correlated: {}'.format(field.object_list)
    # print 'Refining periods and removing aliases'

    for obj in field:
        try:
            t = np.array(obj.lag_timeseries)
            x = np.array(obj.correlations)
            prs = []
            if obj.periods:
                for p, i in zip(obj.periods, obj.peak_indexes):
                    if p < 1.0:
                        # remove periods < 1 day for now
                        continue
                    pr = refine_period(t, x, p)
                    obj.logger.info('Altered {} to {}'.format(p, pr))
                    pr = pr.round(decimals=3)
                    a = obj.ft[i]
                    if (pr, a) not in prs:
                        prs.append((pr, a))
                max_amp = prs[0][1]
                obj.refined_periods = [p for (p, a) in prs if (a > 0.8 * max_amp)]
                obj.cleaned_refined_periods = clear_aliases(obj.refined_periods, 1.0,
                                                            0.1)  # remove 1 day aliases to 0.1 days
                print_list = np.array2string(np.array(obj.cleaned_refined_periods))
                obj.logger.info('Most likely periods {}'.format(print_list))
                pdic[obj.obj] = obj.cleaned_refined_periods
        except Exception as e:
            traceback.print_exc()
            print '{} not processed'.format(obj)
        # try:
        #     obj.plot_data_autocol_ft()
        # except Exception as e:
        #     traceback.print_exc()
        #     print '{} not plotted'.format(obj)
        try:
            obj.save_and_log_object()
        except Exception as e:
            traceback.print_exc()
            print '{} not saved'.format(obj)

        # field.plot_objects_vs_period()

    else:
        print 'Processed Obj IDs {}'.format(field.object_list)

    field = None
    end = time.time()
    diff = end - start
    print 'Worker exited at {}, in {} seconds'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end)), diff)
    return pdic


if __name__ == '__main__':

    ROOT_FILE = '/appch/data/jtb34'

    get_objs_field = NF(fieldname=FIELDNAME, test=CYCLE, root_file=ROOT_FILE)
    field = str(get_objs_field)
    obj_list = get_objs_field.object_list[:10]  # [:NUM_PROC-1]
    del (get_objs_field)

    num_objects = len(obj_list)
    small_list_size = int(np.ceil(float(num_objects) / float(NUM_PROC)))
    #     small_list_size = 10

    fits_file = find_fits_file(FIELDNAME, CYCLE)
    print 'Using file', fits_file

    print "Splitting {} objects into fields of {} size".format(num_objects, small_list_size)

    #     exit()

    explicit_list = []
    for l in chunk_list(obj_list, small_list_size):
        #         instart = time.time()
        #         print 'starting obj_id {} -> {}'.format(l[0], l[-1])
        explicit_list.append((l, fits_file, ROOT_FILE))
    #         worker(l)
    #         inend = time.time()
    #         indiff = inend - instart
    #         print 'objs finished in {} seconds'.format(indiff)

    #     print explicit_list

    start = time.time()
    print "pool initiated at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start)))
    pl = mp.Pool(NUM_PROC)
    pdics = pl.map(wrapped_worker, explicit_list)
    # print pdics
    pdic = {k: v for d in pdics for k, v in d.items()}
    # print pdic

    end = time.time()
    diff = end - start
    print "processing finished at {}, in {} seconds".format(end, diff)

    with open('{}/{}/{}_pdic.json'.format(ROOT_FILE, FIELDNAME, field), 'w') as f:
        json.dump(pdic, f)
