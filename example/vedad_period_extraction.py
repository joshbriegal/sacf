import os
import re
import pandas as pd
import traceback
import logging

file_pattern = re.compile(r'NG(?P<field>\d+[-+]\d+)_(?P<cycle>[\w\d]+?)_')
file_pattern2 = re.compile(r'(?P<field>\d+[-+]\d+)_(?P<obj>\d+?)_LC')

file_location = '/Users/joshbriegal/GitHub/GACF/example/Vedad/ngts_TEST18_variables/'

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('Vedad_GACF_comparison.log', mode='w')
    sh = logging.StreamHandler()
    fh.setFormatter(logging.Formatter())
    sh.setFormatter(logging.Formatter())
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

    logger.info('\n~~~ FINDING VEDAD PERIODS ~~~\n')

    matches = []
    vedad_data = {}
    for file in os.listdir(file_location):
        filename = file_location + file
        match = file_pattern.search(file)
        matches.append(match)
        try:
            field = match.group('field')
            cycle = match.group('cycle')
            df = pd.read_csv(filename, converters={'ID': int, 'p1': float, 'p2': float, 'p3': float,
                                                   'DOUBLE': int, 'a1': float, 'a2': float, 'a3': float})
            df0 = df
            zero_check = df.drop(u'ID', 1) != 0.
            zero_check = zero_check.all(axis=1)
            df = df[zero_check]
            logger.info('Field: {}, Cycle: {}, No. Objs: {} ({} non zero)'.format(field, cycle, len(df0.ID.unique()),
                                                                            len(df.ID.unique())))
            try:
                vedad_data[cycle][field] = df
            except KeyError:
                vedad_data[cycle] = {}
                vedad_data[cycle][field] = df
        except Exception as e:
            traceback.print_exc()
            continue

    logger.info('\n~~~ FINDING GACF PERIODS ~~~\n')

    GACF_periods = {}
    processed_folder = '/Users/joshbriegal/GitHub/GACF/example/processed/'
    period_match = re.compile(r'^interpolated periods: \[(?P<periods>.*)\]')

    for folder in os.listdir(processed_folder):
        match = file_pattern2.search(folder)
        try:
            field = match.group('field')
            obj = int(match.group('obj'))
        except AttributeError as e:
            logger.warning('File {} not processed'.format(folder))
            # traceback.print_exc()
            continue
        else:
            with open(processed_folder + folder + '/peaks.log') as f:
                for l in f:
                    match = period_match.match(l)
                    if match is not None:
                        periods = match.group('periods')
                        periods = re.sub(r',', ' ', periods)
                        periods = periods.split()
                        periods = [float(period) for period in periods]
                        try:
                            GACF_periods[field][obj] = periods
                        except KeyError:
                            GACF_periods[field] = {}
                            GACF_periods[field][obj] = periods

    logger.info('\n~~~ PERIOD COMPARISON ~~~\n')

    for field in GACF_periods.keys():
        logger.info('FIELD {}\n'.format(field))
        for object, periods in GACF_periods[field].items():
            try:
                vedad_object = vedad_data['TEST18'][field][vedad_data['TEST18'][field].ID == object]
            except KeyError as e:
                logger.warning('No match found: {}'.format(e))
            else:
                logger.info('\tObject {}'.format(object))
                logger.info('\t\tVedad periods: {} (double {})'.format(vedad_object[['p1', 'p2', 'p3']].values.tolist(),
                                                             vedad_object['DOUBLE'].values))
                logger.info('\t\tGACF periods: {}'.format(periods))
                logger.info('\n')

    # for object, periods in GACF_periods['0409-1941'].items():
    #     print 'Object ID {}: {}'.format(object, periods)
    # print 'Choosing field 0409-1941, TEST18, objects:'
    # print vedad_data['TEST18']['0409-1941'].ID.unique(), len(vedad_data['TEST18']['0409-1941'].ID.unique())

