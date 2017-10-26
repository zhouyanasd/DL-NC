import time

class Base(object):
    global_time = 0
    print("base: ",global_time)

class implement(Base):

    def print(self):
        for i in range(0,10,1):
            print("implement: ",i,self.global_time)
            Base.global_time = 1


imp = implement()
imp.print()

import numpy as np
class DynamicArray(object):
    def __init__(self, item_type):
        self._data = np.zeros(10, dtype=item_type)
        self._size = 0

    def get_data(self):
        return self._data[:self._size]

    def append(self, value):
        if len(self._data) == self._size:
            self._data = np.resize(self._data, int(len(self._data)*1.25))
        self._data[self._size] = value
        self._size += 1

item_type = np.dtype({
    "names":["id", "x", "y", "z"],
    "formats":["i4", "f8", "f8", "f8"]})

da = DynamicArray(item_type)

for i in range(100):
    da.append((i, i*0.1, i*0.2, i*0.3))

data = da.get_data()

np.zeros(3)