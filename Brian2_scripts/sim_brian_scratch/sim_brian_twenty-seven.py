# ----------------------------------------
# LSM with STDP for JV test
# add Input layer as input and the encoding is transformed into spike trains
# simulation 7--analysis 3
# ----------------------------------------

from brian2 import *
from scipy.optimize import leastsq
import scipy as sp
from sklearn.preprocessing import MinMaxScaler

prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)


# ------define function------------
def lms_train(p0, Zi, Data):
    def error(p, y, args):
        l = len(p)
        f = p[l - 1]
        for i in range(len(args)):
            f += p[i] * args[i]
        return f - y

    Para = leastsq(error, p0, args=(Zi, Data))
    return Para[0]


def lms_test(Data, p):
    l = len(p)
    f = p[l - 1]
    for i in range(len(Data)):
        f += p[i] * Data[i]
    return f


def readout(M, Z):
    n = len(M)
    Data = []
    for i in M:
        Data.append(i)
    p0 = [1] * n
    p0.append(0.1)
    para = lms_train(p0, Z, Data)
    return Data, para


def classification(thea, data):
    data_n = normalization_min_max(data)
    data_class = []
    for a in data_n:
        if a >= thea:
            b = 1
        else:
            b = 0
        data_class.append(b)
    return np.asarray(data_class), data_n


def normalization_min_max(arr):
    arr_n = arr
    for i in range(arr.size):
        x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr_n[i] = x
    return arr_n


def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def load_Data_JV(path="../../Data/jv/train.txt"):
    data = np.loadtxt(path, delimiter=None)
    s = open(path, 'r')
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
    return MinMaxScaler().fit_transform(data), size_d


def get_label(obj, t, size_d, path="../../Data/jv/size.txt"):
    if t == "train":
        data_l = np.loadtxt(path, delimiter=None).astype(int)[1]
    elif t == "test":
        data_l = np.loadtxt(path, delimiter=None).astype(int)[0]
    else:
        raise TypeError("t must be 'train' or 'test'")
    data_l = np.cumsum(data_l)
    data_l_ = np.concatenate(([0], data_l))
    size_d_ = np.asarray(size_d) + 1
    size_d_ = np.concatenate(([0], size_d_))
    label = []
    for i in range(len(data_l_) - 1):
        for j in range(data_l_[i], data_l_[i + 1]):
            for l in range(size_d_[j], size_d_[j + 1]):
                if i == obj:
                    label.append(1)
                else:
                    label.append(0)
    return asarray(label)
