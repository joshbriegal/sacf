import sys
import os
import multiprocessing as mp
import numpy as np
import traceback
import time
import re
import json
from astropy.io import fits
import logging
import logging.handlers
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError

logging.captureWarnings(True)

import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# on server
sys.path.append('/home/jtb34/GitHub/GACF/NGTS/')
from NGTS_Field import NGTSField as NF
from GACF_utils import TIME_CONVERSIONS, refine_period, clear_aliases


# FIELDNAME = 'NG0004-2950'  # 3574 objects, batches of 50
# FIELDNAME = "NG0535-0523"
# FIELDNAME = "NG2058-0248"
# FIELDNAME = "NG2346-3633"
# FIELDNAME = "NG2126-1652"
# FIELDNAME = 'NG0442-3345'  # 5059 objects, batches of 100
# FIELDNAME = 'NG2331-3922'  # xxx objects, batches of 25 (already run 1706) TODO
# FIELDNAME = 'NG0504-3633'  # 6137 objects, batches of 25
# CYCLE = 'CYCLE1807'
# CYCLE = 'CYCLE1807'
# NUM_PROC = min(mp.cpu_count() / 2, 4)
# NUM_PROC = 1
# print 'Found {} CPUs'.format(mp.cpu_count())

NUM_PROC = 32

MAX_OBJECTS_PER_FIELD = 25 #Due to memory restrictions, cannot allocate more than 5GB memory per core => at 100MB per object this is 50 objects.

def init_lock(l):
    global lock
    lock = l

def chunk_list(big_list, small_list_size):
    for i in range(0, len(big_list), small_list_size):
        yield big_list[i:i + small_list_size]


#def find_fits_file(field, cycle, directory=None):
#    if directory is None:
#        directory = os.getcwd()
#    string_pattern = r'^{}_\d+_[0-9\-/]+_{}.fits$'.format(field, cycle)
#    # print string_pattern
#    pattern = re.compile(string_pattern)
#    for f in os.listdir(directory):
#        # print file
#        if re.search(pattern, f) is not None:
#            return f if directory is None else os.path.join(directory, f)
#    raise IOError('File not found in directory {}'.format(directory))

def find_fits_file(directory=None):
    if directory is None:
         directory = os.getcwd()
    string_pattern = r'^(?P<fieldname>\w+?[+-]\w+?)_\d+_[\w\-/,]+_(?P<test>\w+).fits$'
    pattern = re.compile(string_pattern)
    for f in os.listdir(directory):
        match = re.match(pattern, f)
        if match is not None:
            return f if directory is None else os.path.join(directory, f), match.group('fieldname'), match.group('test')

    raise IOError('File not found in directory {}'.format(directory))

def check_processed(directory, fieldname, cycle):
    objs = {}
    dir_name = "_VERSION_{}".format(cycle)
    pattern = re.compile(r'(\d+){}'.format(dir_name))
    patternp = re.compile(r'Most likely periods \[\s*((\d+\.\d+\s*)+)\s*\]')
    patternl = re.compile(r'\d+\.\d+')
    for dirs in os.listdir(directory):
        if dir_name in dirs:
            files = os.listdir(os.path.join(directory, dirs))
            if ('peaks.log' in files) and ('object.json' in files):
                s = open(os.path.join(*[directory, dirs, 'peaks.log'])).read()
                if 'peak periods: [' in s:
                    match = re.match(pattern, dirs)
                    matchp = re.search(patternp, s)
                    if match is not None:
                        if matchp is not None:
                            liststr = matchp.group(1)
                            objs[int(match.group(1))] = [float(i) for i in patternl.findall(liststr)]
                        else:
                            objs[int(match.group(1))] = None
    bad_objs_file = os.path.join(directory, 'bad_objects.json')
    if os.path.isfile(bad_objs_file):
        with open(bad_objs_file, 'r') as f:
            bad_objs = json.load(f)
    else:
        bad_objs = []
    return objs, bad_objs

def wrapped_worker(args):
    
    object_list, fits_file, root_file, fieldname, cycle = args
    
    p_name = 'OBJS_{}-{}.log'.format(object_list[0], object_list[-1])
    plogger = logging.getLogger(p_name)
    pfh = logging.FileHandler(os.path.join(*[root_file, FIELDNAME, p_name]), mode='w')
    pch = logging.StreamHandler()
#    pformatter = logging.Formatter('{asctime} - {name} - {levelname} - {message}')
    pformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    pfh.setFormatter(pformatter)
    pch.setFormatter(pformatter)
    plogger.addHandler(pfh)
    plogger.addHandler(pch)
    plogger.setLevel(logging.INFO)
    
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(pfh)
    warnings_logger.addHandler(pch)
    
    try:
        return worker(object_list, fits_file, root_file, fieldname, cycle,  plogger)
    except:
#        _,_,_,logger = args
        plogger.warning(traceback.format_exc())
        plogger.exception('Subprocess failed, exiting')


def worker(object_list, fits_file, root_file, fieldname, cycle,  plogger):

    start = time.time()
    plogger.info('Worker started at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start))))
    field = NF(fieldname=fieldname, test=cycle, object_list=object_list, root_file=root_file, logger=plogger)
    pdic = {}
    bad_objects = []
    try:
#        lock.acquire()
        plogger.info('Loading field {} from {}'.format(field, fits_file))
        field.load_from_fits(fits_file, obj_ids=object_list, method='fitsio') # astropy runs into memory issues
#        lock.release()
    except Exception as e:
#        plogger.warning(traceback.format_exc())
        plogger.exception('{} unable to be loaded from {}'.format(field, fits_file))
    else:
        bad_objects = [o for o in object_list if o not in field.object_list]
        plogger.info('Loaded {}, Num objects: {}'.format(field, field.num_objects))
        field.get_new_moon_epoch()
        field.get_ratio_estimator()
    for obj in field:
        obj.tbin = 30
        obj.lag_resolution = 30 * TIME_CONVERSIONS['m2d']
        obj.min_lag = 0.
        obj.nsig2keep = 5
        try:
            obj.create_logger()
        except IOError:
            plogger.exception('Unable to create log file for {}'.format(obj))
        try:
            plogger.info('Calculating correlations for {}'.format(obj))
            obj.calculate_periods_from_autocorrelation(calculate_noise_threshold=False)
        except Exception as e:
            plogger.exception('{} not correlated'.format(obj))
            obj.is_ok = False
        else:
            plogger.info("Refining periods and removing aliases for {}".format(obj))
            try:
                fourier_ok = obj.check_fourier_ok(nlim=5)
                if fourier_ok:
                    obj.clean_periods()
                    is_moon = obj.check_moon_detection(field.new_moon_epoch)
                    if is_moon:
                        moon_ok = obj.remove_moon_and_check_ok(field.new_moon_epoch)
                    else:
                        moon_ok = True
                if fourier_ok and moon_ok:
                    # signal_ok = obj.check_signal_significance(epoch=field.new_moon_epoch)
                    signal_ok = obj.check_signal_significance_estimator(field.ratio_estimator, 
                                                                        min_prob=0.7, epoch=field.new_moon_epoch)
                    if signal_ok:
                        pdic[obj.obj] = obj.cleaned_refined_periods
                    else:
                        pdic[obj.obj] = None
                else:
                    pdic[obj.obj] = None
                    obj.cleaned_refined_periods = [] # these periods are bad
            except Exception as e:
    #            plogger.warning(traceback.format_exc())
                plogger.exception('{} not processed'.format(obj))

            try:
                obj.save_and_log_object()
            except Exception as e:
                plogger.exception('{} not saved'.format(obj))
            if not obj.ok:
                bad_objects.append(obj.obj)
            plogger.info('{} saved, removing'.format(obj))
            obj = None # clear object to release memory
            
#    good_objs = [o.obj for o in field if o.ok]
#    bad_objs = [o.obj for o in field if not o.ok]
#    bad_objects += field.remove_bad_objects()
    plogger.info('Processed Obj IDs {}'.format(field.object_list))
#    plogger.info('Removed Obj IDs {}'.format(bad_objs))

    # return bad objects as well to minimise re-processing
    for o in bad_objects:
        pdic[o] = None

    field = None
    end = time.time()
    diff = end - start
    plogger.info('Worker exited at {}, in {} seconds'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end)), diff))
    return pdic


if __name__ == '__main__':
    

    # ROOT_FILE = '/home/jtb34/rds/hpc-work/GACF_OUTPUTS'
    ROOT_FILE = '/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS'
    fits_file, FIELDNAME, CYCLE = find_fits_file()
    
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(os.path.join(*[ROOT_FILE,FIELDNAME,'main.log']), mode='w')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(fh)
    warnings_logger.addHandler(ch)
    
    logger.setLevel(logging.INFO)
    
    logger.info('Using file {}'.format(fits_file))

    get_objs_field = NF(fieldname=FIELDNAME, test=CYCLE, root_file=ROOT_FILE)
    field = str(get_objs_field)
    field_dir = get_objs_field.filename
    # obj_list = get_objs_field.object_list[:10]  # [:NUM_PROC-1]
    get_objbjs_field = None
    with fits.open(fits_file) as f:
        obj_list = map(int, f['CATALOGUE'].data['OBJ_ID'])
    num_objects = len(obj_list)
    
#    processed_objects, bad_objects = check_processed(field_dir, FIELDNAME, CYCLE)
    processed_objects, bad_objects = {}, []
#    logger.info('{}'.format(processed_objects))
    
    logger.info('Already processed {} objects of {} total'.format(len(processed_objects), num_objects))
    logger.info('Will not process {} objects, see bad_objects.json'.format(len(bad_objects)))

#    obj_list = [o for o in obj_list if 33854 < o < 33963]
#    obj_list = [141]
    obj_list = [o for o in obj_list if (o not in processed_objects.keys() and o not in bad_objects)]
    num_objects = len(obj_list)
    pdics = []
    if num_objects > 0:

        small_list_size = min(MAX_OBJECTS_PER_FIELD, int(np.ceil(float(num_objects) / float(NUM_PROC))))
        explicit_list = []
        for i, l in enumerate(chunk_list(obj_list, small_list_size)):
            explicit_list.append((l, fits_file, ROOT_FILE, FIELDNAME, CYCLE))
        num_objects = np.sum([len(e[0]) for e in explicit_list])
        logger.info("Splitting {} objects into {} fields of {} size (on {} cores)".format(num_objects, i+1, small_list_size, NUM_PROC))
    start = time.time()
    
    if num_objects > 0:
        logger.info("Pool initiated at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start))))
        l = mp.Lock()
        with ProcessPool(max_workers=NUM_PROC, initializer=init_lock, initargs=(l,)) as pl:

    #    pl = mp.Pool(NUM_PROC, initializer=init_lock, initargs=(l,))
            future = pl.map(wrapped_worker, explicit_list, timeout=600)
    #        pl.close()
    #        pl.join()
            iterator = future.result()

            # iterate over all results, if a computation timed out log it and continue to the next result
            pcount = 0
            while True:
                pobjs = explicit_list[pcount][0] if pcount < len(explicit_list) else [0]
                pname = 'OBJS_{}-{}'.format(min(pobjs), max(pobjs))
                try:
                    result = next(iterator)
                    pdics.append(result)
                except StopIteration:
                    break  
                except TimeoutError as error:
                    logger.info("{} took longer than {} seconds".format(pname, error.args[1]))
                except ProcessExpired as error:
                    logger.info("{}: {}. Exit code: {}".format(pname, error, error.exitcode))
                except Exception as error:
                    logger.info("{} raised {}" .format(pname, error))
                    logger.info(error.traceback)  # Python's traceback of remote process
                finally:
                    pcount += 1
        pcltime = time.time()
        logger.info("Pool closed at {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pcltime))))
        # print pdics
    pdic_raw = {k: v for d in pdics for k, v in d.items()}
    pdic = {}
    
    for key, value in pdic_raw.iteritems():
        if value is None:
            bad_objects.append(key)
        else:
            pdic[key] = value
    pdic.update(processed_objects)
    # print pdic
    
    with open('{}/{}/{}_pdic.json'.format(ROOT_FILE, FIELDNAME, field).replace(' ', '_'), 'w') as f:
        json.dump(pdic, f)
        
    with open('{}/{}/bad_objects.json'.format(ROOT_FILE, FIELDNAME), 'w') as f:
        json.dump(bad_objects, f)
        
    end = time.time()
    diff = end - start
    logger.info("Processing finished at {}, in {} seconds".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end)), diff))
