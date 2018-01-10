# ----------------------------------------
# LSM with STDP for JV test
# add Input layer as input and the encoding is transformed into spike trains
# simulation 7--analysis 3
# ----------------------------------------

from brian2 import *
from scipy.optimize import leastsq
import scipy as sp
import pandas as pd
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

def get_series_data(data_frame, pattern_interval, is_order = True):
    pass

# -----parameter setting-------
data_train = load_Data_JV(t='train',path_value="../../Data/jv/train.txt")
data_test = load_Data_JV(t='test',path_value="../../Data/jv/test.txt")

pattern_interval = 400
data_train_s = get_series_data(data_train,pattern_interval)
data_test_s = get_series_data(data_test,pattern_interval)

duration_train = len(data_train) * defaultclock.dt
duration_test = len(data_test)*defaultclock.dt

obj = 1
n = 20

threshold = 0.5

equ_in = '''
dv/dt = (I-v) / (1.5*ms) : 1 (unless refractory)
I = stimulus(t,i) : 1
'''

equ = '''
r : 1
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(1.5*ms*r) : 1
dh/dt = (-h)/(1.45*ms*r) : 1
I = tanh(g-h)*20: 1
'''

equ_read = '''
dg/dt = (-g)/(1.5*ms) : 1 
dh/dt = (-h)/(1.45*ms) : 1
I = tanh(g-h)*20 : 1
'''

model_STDP = '''
w : 1
wmax : 1
wmin : 1
Apre : 1
Apost = -Apre * taupre / taupost * 1.2 : 1
taupre : second
taupost : second
dapre/dt = -apre/taupre : 1 (clock-driven)
dapost/dt = -apost/taupost : 1 (clock-driven)
'''

on_pre = '''
h+=w
g+=w
'''

on_pre_STDP = '''
h+=w
g+=w
apre += Apre
w = clip(w+apost, wmin, wmax)
'''

on_post_STDP = '''
apost += Apost
w = clip(w+apre, wmin, wmax)
'''

# -----simulation setting-------
data_pre, label_pre = Tri_function(pre_train_duration, pattern_duration = pattern_duration,
                                   pattern_interval = pattern_interval, obj=obj)
data, label = Tri_function(duration + duration_test, pattern_duration = pattern_duration,
                           pattern_interval = pattern_interval)
Time_array = TimedArray(data, dt=defaultclock.dt)

Time_array_pre = TimedArray(data_pre, dt=defaultclock.dt)

Input = NeuronGroup(1, equ_in, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                    name = 'neurongroup_input')

G = NeuronGroup(n, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                name='neurongroup')

G2 = NeuronGroup(int(n / 4), equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                 name='neurongroup_1')

G_lateral_inh = NeuronGroup(1, equ, threshold='v > 0.20', reset='v = 0', method='euler', refractory=1 * ms,
                            name='neurongroup_la_inh')

G_readout = NeuronGroup(n, equ_read, method='euler', name='neurongroup_read')

S = Synapses(Input, G, 'w : 1', on_pre = on_pre ,method='linear', name='synapses')

S2 = Synapses(G2, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_1')

S3 = Synapses(Input, G_lateral_inh, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_2')

S5 = Synapses(G, G2, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_4')

S4 = Synapses(G, G, model_STDP, on_pre=on_pre_STDP, on_post=on_post_STDP, method='linear', name='synapses_3')

S6 = Synapses(G_lateral_inh, G, 'w : 1', on_pre=on_pre, method='linear', name='synapses_5')

S_readout = Synapses(G, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

# -------network topology----------