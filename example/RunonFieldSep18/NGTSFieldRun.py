import sys
import os
import multiprocessing as mp
import numpy as np
import traceback
import time

import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# on server
sys.path.append('/home/jtb34/GitHub/GACF')
from NGTS.NGTS_Field import NGTSField as NF

# FIELDNAME = 'NG0004-2950'  # 3574 objects, batches of 50
# FIELDNAME = "NG0535-0523"
# FIELDNAME = "NG2058-0248"
# FIELDNAME = "NG2346-3633"
FIELDNAME = "NG2126-1652"
# FIELDNAME = 'NG0442-3345'  # 5059 objects, batches of 100
# FIELDNAME = 'NG2331-3922'  # xxx objects, batches of 25 (already run 1706) TODO
# FIELDNAME = 'NG0504-3633' # 6137 objects, batches of 25
CYCLE = 'CYCLE1807'
# CYCLE = 'CYCLE1807'
NUM_PROC = mp.cpu_count() - 1


# NUM_PROC = 16


def chunk_list(big_list, small_list_size):
    for i in range(0, len(big_list), small_list_size):
        yield big_list[i:i + small_list_size]


def worker(object_list):
    start = time.time()
    print 'Worker started at {}'.format(start)
    field = NF(fieldname=FIELDNAME, test=CYCLE, object_list=object_list, root_file='/appch/data/jtb34')
    print field, 'Num objects: {}'.format(field.num_objects)
    try:
        field.get_object_lc_bulk()
        field.calculate_periods_from_autocorrelation()
        field.save_object_plots_and_objects()
        # field.plot_objects_vs_period()
    except Exception as e:
        traceback.print_exc()
        print 'Obj IDs not processed: {}'.format(field.object_list)
    else:
        print 'Processed Obj IDs {}'.format(field.object_list)
    field = None
    end = time.time()
    diff = end - start
    print 'Worker exited at {}, in {} seconds'.format(end, diff)


if __name__ == '__main__':

    get_objs_field = NF(fieldname=FIELDNAME, test=CYCLE, root_file='/appch/data/jtb34')
    obj_list = get_objs_field.object_list  # [:NUM_PROC-1]
    del (get_objs_field)

    num_objects = len(obj_list)
    small_list_size = int(np.ceil(float(num_objects) / float(NUM_PROC)))
    #     small_list_size = 10

    print "Splitting {} objects into fields of {} size".format(num_objects, small_list_size)

    #     exit()

    explicit_list = []
    for l in chunk_list(obj_list, small_list_size):
        #         instart = time.time()
        #         print 'starting obj_id {} -> {}'.format(l[0], l[-1])
        explicit_list.append(l)
    #         worker(l)
    #         inend = time.time()
    #         indiff = inend - instart
    #         print 'objs finished in {} seconds'.format(indiff)

    #     print explicit_list

    start = time.time()
    print "pool initiated at {}".format(start)
    pl = mp.Pool(NUM_PROC)
    pl.map(worker, explicit_list)

    end = time.time()
    diff = end - start
    print "processing finished at {}, in {} seconds".format(end, diff)
