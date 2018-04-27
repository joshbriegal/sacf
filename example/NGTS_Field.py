"""
PyObject per NGTS field which contains a list of NGTS objects & associated PyObjects
"""

from ngtsio import ngtsio
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


def worker(arg):
    """
    Used for multiprocessing of reading output data from NGTSio
    :param arg: Tuple (object, NGTSio output dictionary, bin, delete_unbinned)
    :return: None
    """
    obj, dic, bin, delete_unbinned = arg
    obj.get_data_from_dict(dic)
    if bin:
        obj.bin_data(delete_unbinned=delete_unbinned)
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


class Field(object):

    def __init__(self, fieldname, test="CYCLE1706", root_file=None, nObj=None, initialise_objects=True):
        self.fieldname = fieldname
        self.test = test
        self.object_list = None
        self.num_objects = None
        self.objects = []
        self.filename = os.path.join(os.getcwd() if root_file is None else root_file, self.fieldname)
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)
        try:
            self.get_objects(initialise_objects=initialise_objects, nObj=nObj)
        except TypeError:
            pass

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

    def remove_bad_objects(self):
        self.objects = [obj for obj in self.objects if obj.ok]
        self.num_objects = len(self.objects)

    def get_objects(self, initialise_objects=True, nObj=None, **kwargs):
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
            self.objects = list(pool.map(worker, [(obj, deepcopy(dic), bin, delete_unbinned) for obj in self.objects]))
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
                               save_if_true_else_return=True):
        if calculate:
            self.calculate_periods_from_autocorrelation(calculate_noise=signal_to_noise)
        if fig_ax_tuple is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax_tuple

        for i, obj in enumerate(self):
            periods = obj.periods
            size = obj.peak_signal_to_noise
            y = np.linspace(i, i, len(periods))
            ax.scatter(periods, y, s=size, c='k')

        if fig_ax_tuple is None:
            ax.set_ylabel('Object no.')
            ax.set_xlabel('Period (days)')
            ax.set_xscale('log')
            ax.xaxis.grid(True, linestyle='--', alpha=0.5, which='both', lw=0.1)
            ax.set_xlim(left=0)
            ax.set_title('Objects in Field {} and GACF periods'.format(self.fieldname))
            ax.set_yticks(np.linspace(0, self.num_objects - 1, self.num_objects))
            ax.set_yticklabels(['obj_id {}'.format(obj.obj) for obj in self], fontdict={'fontsize': 6})

        ax.set_yticks(np.append(ax.get_yticks(), np.linspace(0, self.num_objects - 1, self.num_objects)))
        ax.set_yticklabels(np.append(ax.get_yticklabels()[1], ['obj_id {}'.format(obj.obj) for obj in self]),
                           fontdict={'fontsize': 6})

        if save_if_true_else_return:
            fig.savefig(os.path.join(self.filename, 'Objects vs Period.pdf'))
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



if __name__ == '__main__':
    # from time import time
    #
    field = Field('NG2331-3922', nObj=10, root_file='/data/jtb34/')
    # field2 = Field('NG2331-3922', nObj=10)
    # obj_JJ = NGTSObject(obj=9138, field='NG2331-3922', test='CYCLE1706')
    # obj_JJ.get_binned_data()
    #
    # start = time()
    # field.get_object_lc(delete_unbinned=True)
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
    for obj in field:
        if obj.obj == "000334":
            print "\t\t *** Processing", obj, " ***\n"
            obj.get_binned_data()
            obj.calculate_noise_threshold()
            obj.calculate_periods_from_autocorrelation()
            obj.plot_data_autocol_ft()
            obj.plot_autocol()
            obj.plot_ft()
            obj.plot_phase_folded_lcs()
            obj.save_and_log_object()
            print "\t\t *** ", obj, "Processed *** \n"

    # field.pickle_field()
