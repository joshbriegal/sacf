"""
PyObject per NGTS field which contains a list of NGTS objects & associated PyObjects
"""
import warnings

try:
    from ngtsio import ngtsio

    ngtsio_import = True
except ImportError:
    warnings.warn("ngtsio not imported")
    ngtsio_import = False
from NGTS_Object import NGTSObject, return_object_from_json_string
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
from copy import deepcopy
import os
from GACF_utils import get_ngts_data, ClassEncoder, NGTSObjectDecoder
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
import re
import matplotlib as mpl


def get_data_worker(arg):
    """
    Used for multiprocessing of reading output data from NGTSio
    :param arg: Tuple (object, NGTSio output dictionary, bin, delete_unbinned)
    :return: None
    """
    obj, dic, bin, delete_unbinned = arg
    obj.get_data_from_dict(dic)
    if not obj.ok:
        obj = None
        return obj
    if bin:
        obj.bin_data(delete_unbinned=delete_unbinned)
    return obj

def object_call_worker(arg):
    """
    Used for multiprocessing of reading output data from NGTSio and calculating periods
    :param arg: Tuple (object, NGTSio output dictionary, bin, delete_unbinned)
    :return: None
    """
    obj = get_data_worker(arg)
    if obj is None:
        return obj
    else:
        obj.calculate_periods_from_autocorrelation(calculate_noise_threshold=True)
        obj.save_and_log_object(clear_unneeded_data=True)
    return obj


def return_field_from_json_str(json_str):
    if isinstance(json_str, str):
        dic = json.loads(json_str, cls=NGTSObjectDecoder)
    elif isinstance(json_str, dict):
        dic = json_str
    else:
        raise TypeError("Input must be string or dict")
    if 'objects' not in dic.keys():
        raise ValueError('No object list found')
    for i, obj_dic in enumerate(dic['objects']):
        print obj_dic
        dic['objects'][i] = return_object_from_json_string(obj_dic)
    tfield = Field('temp')
    tfield.__dict__ = dic
    return tfield

def return_field_from_object_directory(root_directory, fieldname, test=None):
    if os.path.exists(os.path.join(root_directory, fieldname)):
        filename = os.path.join(root_directory, fieldname)
    else:
        filename = root_directory
    field = Field(fieldname=fieldname, root_file=root_directory, nObj=0, test=test)

    object_file_pattern = re.compile(r'^(?P<obj>\d)+_VERSION_(?P<test>.+?)$')
    object_json_filename = 'object.json'
    for i, obj_file in enumerate(os.listdir(filename)):
        match = object_file_pattern.search(obj_file)
        if match is not None:
            try:
                obj_id = match.group('obj')
                test = match.group('test')
                if i == 0:
                    field.test = test
                if test != field.test:
                    continue
            except KeyError:
                continue
            obj_file = os.path.join(filename, obj_file)
            if object_json_filename in os.listdir(obj_file):
                with open(os.path.join(obj_file, object_json_filename), 'r') as jso:
                    obj = return_object_from_json_string(jso.read())
                    obj.remove_logger()
                    field.objects.append(obj)
                    obj = None
                    jso.close()
            else:
                continue
    field.num_objects = len(field.objects)
    return field


class Field(object):

    def __init__(self, fieldname, test="CYCLE1706", root_file=None, nObj=None, object_list=None,
                 initialise_objects=True, log_to_console=False):
        self.fieldname = fieldname
        self.test = test
        self.object_list = object_list
        self.num_objects = None
        self.objects = []
        self.filename = os.path.join(os.getcwd() if root_file is None else root_file, self.fieldname)
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)
        try:
            self.get_objects(initialise_objects=initialise_objects, nObj=nObj)
        except TypeError:
            pass

        self.create_logger(to_console=log_to_console)

    def __str__(self):
        return "NGTS Field " + self.fieldname

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, i):
        return self.objects[i]

    def __iter__(self):
        self.it = 0
        return self

    def __next__(self):
        if self.it < self.num_objects:
            obj = self.objects[self.it]
            self.it += 1
            return obj
        else:
            raise StopIteration

    next = __next__

    def __call__(self, bin=True, delete_unbinned=True):
        if len(self.objects) == 0:
            self.initialise_objects()
        dic = get_ngts_data(fieldname=self.fieldname, obj_id=self.object_list, ngts_version=self.test, return_dic=True)
        pool = Pool(cpu_count())
        objects = pool.imap(object_call_worker, [(obj, deepcopy(dic), bin, delete_unbinned) for obj in self.objects])
        # for o in self.objects:
            # self.logger.info("\t\t *** Processing {} ***".format(o))
            # object_call_worker((o, dic, bin, delete_unbinned))
        pool.close()
        pool.join()
        try:
            self.objects = [obj for obj in objects if obj is not None]
        except TypeError:
            self.objects = list(objects)
        self.num_objects = len(self.objects)
        # self.plot_objects_vs_period()
        

    def create_logger(self, logger=None, to_console=False):
        if logger is None:
            fh = logging.FileHandler(os.path.join(self.filename, 'Log.txt'), 'w')
            fh.setFormatter(logging.Formatter())  # default formatter
            logger = logging.getLogger(str(self))
            logger.addHandler(fh)
            if to_console:
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter())
                logger.addHandler(sh)
            logger.setLevel(logging.INFO)
        self.logger = logger
        return

    def remove_bad_objects(self):
        self.objects = [obj for obj in self.objects if obj.ok]
        self.num_objects = len(self.objects)

    def get_objects(self, initialise_objects=True, nObj=None, override_object_list=False, **kwargs):
        if not ngtsio_import or nObj == 0:
            return
        if self.object_list is None or override_object_list:
            dic = ngtsio.get(self.fieldname, self.test, ["OBJ_ID"], silent=True)
            self.object_list = dic["OBJ_ID"] if nObj is None else dic["OBJ_ID"][:nObj]
        self.num_objects = len(self.object_list)
        if initialise_objects:
            self.initialise_objects(*kwargs)
        return

    def initialise_objects(self, **kwargs):
        if self.object_list is None:
            raise ValueError("No object list to initialise from")
        for obj_id in self.object_list:
            self.objects.append(NGTSObject(obj=obj_id, field=self.fieldname, test=self.test,
                                           field_filename=self.filename, *kwargs))
        return

    def get_object_lc(self, obj_ids=None, bin=True, delete_unbinned=True):

        if len(self.objects) == 0:
            self.initialise_objects()
        for obj in tqdm(self.objects if obj_ids is None else [o for o in self.objects if o in obj_ids]):
            if bin:
                obj.get_binned_data(delete_unbinned=delete_unbinned)
            else:
                obj.get_data()
        self.remove_bad_objects()

    def get_object_lc_multi(self, obj_ids=None, bin=True, delete_unbinned=True, use_multiprocessing=True):

        if len(self.objects) == 0:
            self.initialise_objects()
        dic = get_ngts_data(fieldname=self.fieldname, obj_id=self.object_list if
        obj_ids is None else obj_ids, ngts_version=self.test, return_dic=True)
        if use_multiprocessing:
            pool = Pool(cpu_count())
            self.objects = list(pool.map(get_data_worker, [(obj, deepcopy(dic), bin, delete_unbinned) for obj in self.objects]))
            pool.close()
            pool.join()
        else:
            for obj in self.objects:
                obj.get_data_from_dict(dic)
                if bin:
                    obj.bin_data(delete_unbinned=delete_unbinned)
        self.remove_bad_objects()

    def pickle_field(self, pickle_file=None):
        if pickle_file is None:
            pickle_file = os.path.join(self.filename, '{}.pkl'.format(self))

        pickle.dump(self, open(pickle_file, 'w'))
        return

    def field_to_json(self):
        return json.dumps(self, cls=ClassEncoder)

    def calculate_noise_thresholds(self, **kwargs):
        for obj in self:
            obj.calculate_noise_threshold(**kwargs)
        return

    def calculate_autocorrelations(self, **kwargs):
        for obj in self:
            obj.calculate_autocorrelation(**kwargs)
        return

    def calculate_periods_from_autocorrelation(self, calculate_noise=True, **kwargs):
        for obj in self:
            obj.calculate_periods_from_autocorrelation(calculate_noise_threshold=calculate_noise, **kwargs)
        return

    def plot_objects_vs_period(self, calculate=False, signal_to_noise=False, fig_ax_tuple=None,
                               save_if_true_else_return=True, interactive=False):
        if calculate:
            self.calculate_periods_from_autocorrelation(calculate_noise=signal_to_noise)
        if fig_ax_tuple is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig, ax = fig_ax_tuple

        max_signal_to_noise = max([max(obj.peak_signal_to_noise) for obj in self if np.array(obj.peak_signal_to_noise).size > 0])
        min_signal_to_noise = min([min(obj.peak_signal_to_noise) for obj in self if np.array(obj.peak_signal_to_noise).size > 0])
        min_obj_id = np.inf
        max_obj_id = 0

        if fig_ax_tuple is None:
            c_min = 1
            c_max = 2  # max_signal_to_noise
            axc = fig.add_axes([0.95, 0.1, 0.01, 0.8])
            norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
            cb = mpl.colorbar.ColorbarBase(axc, norm=norm, cmap=mpl.cm.viridis, orientation='vertical')
            cb.set_clim(vmin=c_min, vmax=c_max)
            cb.set_label("Signal to Noise")
        
        for i, obj in enumerate(self):
            periods = obj.periods
            color = obj.peak_signal_to_noise

            obj_id = int(obj.obj)
            if obj_id > max_obj_id:
                max_obj_id = obj_id
            elif obj_id < min_obj_id:
                min_obj_id = obj_id

            if np.array(color).size > 0:
                size = np.interp(color, (min_signal_to_noise, max_signal_to_noise), (0.1, 20))
            else:
                size = color
            y = np.linspace(obj_id, obj_id, len(periods))
            # ax.scatter(periods, y, s=size, c='k')
            ax.scatter(periods, y, c=color, cmap=mpl.cm.viridis, s=size, norm=norm)
            
        if fig_ax_tuple is None:
            ax.set_ylabel('Object no.')
            ax.set_xlabel('Period (days)')
            ax.set_xscale('log')
            ax.xaxis.grid(True, linestyle='--', alpha=0.5, which='both', lw=1)
            ax.set_xlim(left=0)
            ax.set_ylim([min_obj_id*0.8, max_obj_id + (min_obj_id*0.2)])
            ax.set_title('Objects in Field {} and GACF periods'.format(self.fieldname))
            # ax.set_yticks(np.linspace(0, self.num_objects - 1, self.num_objects))
            # ax.set_yticklabels(['obj_id {}'.format(obj.obj) for obj in self], fontdict={'fontsize': 6})

            

            # ax.set_facecolor('lightgray')

            # ax.set_yticks(np.append(ax.get_yticks(), np.linspace(0, self.num_objects - 1, self.num_objects)))
            # ax.set_yticklabels(np.append(ax.get_yticklabels()[1], ['obj_id {}'.format(obj.obj) for obj in self]),
                               # fontdict={'fontsize': 6})

        if save_if_true_else_return:
            if interactive:
                plt.show()
            else:
                fig.savefig(os.path.join(self.filename, 'Objects vs Period.png'))
                plt.close(fig)
                return
        else:
            return fig, ax

    def save_object_plots_and_objects(self, remove_object_once_processed=False):
        for i, obj in enumerate(self):
            obj.plot_data_autocol_ft()
            obj.plot_autocol()
            obj.plot_ft()
            obj.plot_phase_folded_lcs()
            obj.save_and_log_object()
            if remove_object_once_processed:
                self.objects[i] = None

        self.objects = [obj for obj in self.objects if obj is not None]
        self.num_objects = len(self.objects)

    def save_field(self):
        self.logger = None  # to allow JSON serialisation must remove logger
        jsf = self.field_to_json()
        json_file = open(os.path.join(self.filename, 'field.json'), 'w')
        json_file.write(jsf)
        json_file.close()

    def return_object_periods_dict(self):
        periods_dict = {}
        for obj in self:
            out_dict = {"periods": obj.periods, "signal_to_noise": obj.peak_signal_to_noise}
            periods_dict[str(obj)] = out_dict
        return periods_dict
            


def chunk_list(big_list, small_list_size):
    for i in range(0, len(big_list), small_list_size):
        yield big_list[i:i+small_list_size]


if __name__ == '__main__':
    # from time import time
    #

    dic_file = open('/data/jtb34/output_dictionary.dat','w')

    field = Field('NG2331-3922', root_file='/data/jtb34/', log_to_console=True)
    object_list = field.object_list

    # get already processed objects
    f = open('already_processed.txt', 'r')
    obj_str = f.read()
    f.close()

    processed = obj_str.split(',')
    
    object_list = [o for o in object_list if o not in processed]

    period_dict = {}
    
    # batch objects within field to prevent memory overflow    
    
    for object_sublist in chunk_list(object_list, 50):
        print "start", object_sublist
        field = Field('NG2331-3922', root_file='/data/jtb34', log_to_console=True, object_list=object_sublist)
        print "in object", field.object_list, field.objects
        field()
        period_dict.update(field.return_object_periods_dict())
        
    
    dic_file.write(str(period_dict))


    # field.save_field()
    # field2 = Field('NG2331-3922', nObj=10)
    # obj_JJ = NGTSObject(obj=9138, field='NG2331-3922', test='CYCLE1706')
    # obj_JJ.get_binned_data()
    #
    # start = time()
    # field.get_object_lc_multi(delete_unbinned=True)
    # end = time()
    # field2.get_object_lc(delete_unbinned=False)
    # end2 = time()
    #
    # print "Time get object lc multi: {}".format(end - start)
    # print "Time get object lc: {}".format(end2 - end)

    # field = return_field_from_json_str(open('field.json', 'r').read())
    # field.calculate_periods_from_autocorrelation()
    # field.save_object_plots_and_objects()
    # field.plot_objects_vs_period()
    # for obj in field:
        # if obj in ["000334", "000321"]:
            # print "\t\t *** Processing", obj, " ***\n"
            # obj.get_binned_data()
#            obj.calculate_noise_threshold()
            # obj.calculate_periods_from_autocorrelation()
            # obj.plot_data_autocol_ft()
            # obj.plot_autocol()
            # obj.plot_ft()
            # obj.plot_phase_folded_lcs()
            # obj.save_and_log_object()
            # print "\t\t *** ", obj, "Processed *** \n"

    # field.objects = [obj for obj in field if obj.num_observations_binned is not None]
    # # print "updating", field[1], " timeseries_binned"
    # # field[1].timeseries_binned = field[0].timeseries_binned[:field[1].num_observations_binned]
    # # field[1].flux_binned = field[0].flux_binned[:field[1].num_observations_binned]
    # field[1].calculate_autocorrelation()


    # field.pickle_field()
