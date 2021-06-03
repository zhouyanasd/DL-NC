import numpy as np
import math
import brian2



class BaseFunctions():
    """
    Some basic functions for the simulations

    """

    def data_batch(self, data, n_batch):
        batches = []
        n_data = math.ceil(data.shape[0] / n_batch)
        for i in range(n_batch):
            sub_data = data[i * n_data:(i + 1) * n_data]
            batches.append(sub_data)
        return batches

    def sub_list(self, l, s):
        return [l[x] for x in s]

    def get_sub_dict(self, _dict, *keys):
        return {key:value for key,value in _dict.items() if key in keys}

    def change_dict_key(self, _dict, key1, key2):
        _dict[key2] = _dict.pop(key1)

    def adapt_scale(self, scale, parameter):
        return scale[0] + parameter * abs(scale[1] - scale[0])

    def np_extend(self, a, b, axis=0):
        if a is None:
            shape = list(b.shape)
            shape[axis] = 0
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b, axis)

    def np_append(self, a, b):
        shape = list(b.shape)
        shape.insert(0, -1)
        if a is None:
            a = np.array([]).reshape(tuple(shape))
        return np.append(a, b.reshape(tuple(shape)), axis=0)