import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ClassEncoder(NumpyEncoder):

    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super(ClassEncoder, self).default(obj)


class NGTSObjectDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        for key in obj.keys():
            obj[key] = recursively_create_nd_array_from_list(obj[key])
        return obj


def recursively_create_nd_array_from_list(list_in):
    if isinstance(list_in, list):
        list_in = np.array(list_in)
        for i, item in enumerate(list_in):
            list_in[i] = recursively_create_nd_array_from_list(item)

    return list_in
