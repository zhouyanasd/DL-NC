# ----------------------------------------
# LSM with STDP for JV test
# add Input layer as input and the encoding is transformed into spike trains
# simulation 7--analysis 3
# ----------------------------------------

import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

np.random.seed(100)

# ------define function------------
def normalization_min_max(arr):
    arr_n = arr
    for i in range(arr.size):
        x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr_n[i] = x
    return arr_n


def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def load_Data_JV(t, path_value="../../Data/jv/train.txt", path_label="../../Data/jv/size.txt"):
    if t == "train":
        label = np.loadtxt(path_label, delimiter=None).astype(int)[1]
    elif t == "test":
        label = np.loadtxt(path_label, delimiter=None).astype(int)[0]
    else:
        raise TypeError("t must be 'train' or 'test'")
    data = np.loadtxt(path_value, delimiter=None)
    data = MinMaxScaler().fit_transform(data)
    s = open(path_value, 'r')
    i = -1
    size_d = []
    while True:
        lines = s.readline()
        i += 1
        if not lines:
            break
        if lines == '\n':  # "\n" needed to be added at the end of the file
            i -= 1
            size_d.append(i)
            continue
    size_d = np.asarray(size_d) + 1
    size_d = np.concatenate(([0], size_d))
    data_list = [data[size_d[i]:size_d[i + 1]] for i in range(len(size_d) - 1)]
    label_list = []
    j = 0
    for n in label:
        label_list.extend([j] * n)
        j += 1
    data_frame = pd.DataFrame({'value': pd.Series(data_list), 'label': pd.Series(label_list)})
    return data_frame

#----------------------------------

data = load_Data_JV('test',path_value="../../Data/jv/test.txt")

print(data)