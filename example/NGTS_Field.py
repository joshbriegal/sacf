"""
PyObject per NGTS field which contains a list of NGTS objects & associated PyObjects
"""

from ngtsio import ngtsio
from NGTS_object import NGTSObject
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
from copy import deepcopy
import os
from GACF_utils import get_ngts_data


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


class Field(object):

    def __init__(self, fieldname, test="CYCLE1706", nObj=None, initialise_objects=True):
        self.fieldname = fieldname
        self.test = test
        self.object_list = None
        self.num_objects = None
        self.objects = []
        self.filename = os.path.join(os.getcwd(), self.fieldname)
        self.get_objects(initialise_objects=initialise_objects, nObj=nObj)

        

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


if __name__ == '__main__':
    from time import time

    field = Field('NG2331-3922', nObj=10)
    field2 = Field('NG2331-3922', nObj=10)
    obj_JJ = NGTSObject(obj=9138, field='NG2331-3922', test='CYCLE1706')
    obj_JJ.get_binned_data()

    start = time()
    field.get_object_lc_multi(delete_unbinned=False)
    end = time()
    field2.get_object_lc(delete_unbinned=False)
    end2 = time()

    print "Time get object lc multi: {}".format(end - start)
    print "Time get object lc: {}".format(end2 - end)

    # field.pickle_field()

