# ----------------------------------------
# LSM without STDP for MNIST test
# add neurons to readout layer for multi-classification(one-versus-the-rest)
# using softmax(logistic regression)
# input layer is changed to 781*1 with encoding method
# change the LSM structure according to Maass paper
# ----------------------------------------

from brian2 import *
from brian2tools import *
import scipy as sp
import struct
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)

# ------define function------------
def softmax(z):
    return np.array([(np.exp(i) / np.sum(np.exp(i))) for i in z])

def train(X, Y, P):
    a = 0.0001
    max_iteration = 10000
    time = 0
    while time < max_iteration:
        time += 1
        P = P + X.T.dot(Y - softmax(X.dot(P))) * a
    return P

def lms_test(Data, p):
    one = np.ones((Data.shape[1], 1)) #bis
    X = np.hstack((Data.T, one))
    return X.dot(p)

def readout(M, Y):
    one = np.ones((M.shape[1], 1))
    X = np.hstack((M.T, one))
    P = np.random.rand(X.shape[1],Y.T.shape[1])
    para = train(X, Y.T, P)
    return para

def normalization_min_max(arr):
    arr_n = arr
    for i in range(arr.size):
        x = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr_n[i] = x
    return arr_n


def mse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))


def label_to_obj(label, obj):
    temp = []
    for a in label:
        if a == obj:
            temp.append(1)
        else:
            temp.append(0)
    return np.asarray(temp)


def one_versus_the_rest(label, *args, **kwargs):
    obj = []
    for i in args:
        temp = label_to_obj(label, i)
        obj.append(temp)
    try:
         for i in kwargs['selected']:
            temp = label_to_obj(label, i)
            obj.append(temp)
    except KeyError:
        pass
    return np.asarray(obj)


def trans_max_to_label(results):
    labels = []
    for result in results:
        labels.append(np.argmax(result))
    return labels


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


def get_states(input, interval, duration, sample):
    n = int(duration / interval)
    step = int(interval / sample / defaultclock.dt)
    interval_ = int(interval / defaultclock.dt)
    temp = []
    for i in range(n):
        sum = np.sum(input[:, i * interval_: (i + 1) * interval_: step], axis=1)
        temp.append(sum)
    return MinMaxScaler().fit_transform(np.asarray(temp).T)


def load_Data_MNIST(n, path_value, path_label):
    with open(path_value, 'rb') as f1:
        buf1 = f1.read()
    with open(path_label, 'rb') as f2:
        buf2 = f2.read()

    image_index = 0
    image_index += struct.calcsize('>IIII')
    im = []
    for i in range(n):
        temp = struct.unpack_from('>784B', buf1, image_index)
        im.append(np.reshape(temp, (1, 784)))
        image_index += struct.calcsize('>784B')

    label_index = 0
    label_index += struct.calcsize('>II')
    label = np.asarray(struct.unpack_from('>' + str(n) + 'B', buf2, label_index))

    f = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    df = pd.DataFrame({'value': pd.Series(im).apply(f), 'label': pd.Series(label)})
    return df


def encoding_latency_MNIST(analog_data, n, duration, min = 0, max = np.pi, *args):
    def encoding_cos(x, n, A):
        encoding = []
        for i in range(int(n)):
            trans_cos = np.round(A*(np.cos(x+np.pi*np.random.rand())+1)).clip(0,2*A-1)
            coding = [([0] * trans_cos.shape[1]) for i in range(2*A*trans_cos.shape[0])]
            index_0 = 0
            for p in trans_cos:
                index_1 = 0
                for q in p:
                    coding[int(q)][index_1+A*index_0] = 1
                    index_1 += 1
                index_0 += 1
            encoding.extend(coding)
        return np.asarray(encoding)
    f = lambda x: (max-min)*(x - np.min(x)) / (np.max(x) - np.min(x))
    return analog_data.apply(f).apply(encoding_cos, n = n, A = duration)


def get_series_data(n, data_frame, duration, is_order=True, *args, **kwargs):
    try:
        obj = kwargs['obj']
    except KeyError:
        obj = np.arange(10)
    if not is_order:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)].sample(frac=1).reset_index(drop=True)
    else:
        data_frame_obj = data_frame[data_frame['label'].isin(obj)]
    data_frame_s = []
    for value in encoding_latency_MNIST(data_frame_obj['value'][:n], 3, 5):
        for data in value:
            data_frame_s.append(list(data))
        interval = duration - value.shape[0]
        if interval > 30:
            data_frame_s.extend([[0] * 784] * interval)
        else:
            raise Exception('duration is too short')
    data_frame_s = np.asarray(data_frame_s)
    label = data_frame_obj['label'][:n]
    return data_frame_s, label

# -----parameter setting-------
duration = 100
N_train = 100
N_test = 100
Dt = defaultclock.dt
pre_train_loop = 0
sample = 2

n = 108
i_EE = 30
i_EI = 60
i_IE = -19
i_II = -19
i_inE = 18
i_inI = 9


df_train = load_Data_MNIST(60000, '../../Data/MNIST_data/train-images.idx3-ubyte',
                               '../../Data/MNIST_data/train-labels.idx1-ubyte')
df_test = load_Data_MNIST(10000, '../../Data/MNIST_data/t10k-images.idx3-ubyte',
                               '../../Data/MNIST_data/t10k-labels.idx1-ubyte')

data_train_s, label_train = get_series_data(N_train, df_train, duration, False)
data_test_s, label_test = get_series_data(N_test, df_test, duration, False)

duration_train = len(data_train_s) * Dt
duration_test = len(data_test_s) * Dt

equ_in = '''
I = stimulus(t,i) : 1
'''

equ = '''
dv/dt = (I-v) / (3*ms) : 1 (unless refractory)
dg/dt = (-g)/(0.3*ms) : 1
dh/dt = (-h)/(0.25*ms) : 1
I = (g-h)+13.5: 1
'''

equ_read = '''
dg/dt = (-g)/(0.3*ms) : 1 
dh/dt = (-h)/(0.25*ms) : 1
I = (g-h): 1
'''

on_pre = '''
h+=w
g+=w
'''


# -----simulation setting-------
Time_array_train = TimedArray(data_train_s, dt=Dt)

Time_array_test = TimedArray(data_test_s, dt=Dt)

Input = NeuronGroup(784, equ_in, threshold='I > 0', reset='I = 0', method='linear', refractory=0 * ms,
                    name = 'neurongroup_input')

G_ex = NeuronGroup(n, equ, threshold='v > 15', reset='v = 13.5', method='euler', refractory=0.3 * ms,
                name='neurongroup')

G_in = NeuronGroup(int(n/4), equ, threshold='v > 15', reset='v = 13.5', method='euler', refractory=0.2 * ms,
                name='neurongroup')

G_readout = NeuronGroup(n, equ_read, method='euler', name='neurongroup_read')

S_inE = Synapses(Input, G_ex, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_inE')

S_inI = Synapses(Input, G_in, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_inI')

S_EE = Synapses(G_ex, G_ex, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_EE')

S_EI = Synapses(G_ex, G_in, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_EI')

S_IE = Synapses(G_in, G_ex, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_IE')

S_II = Synapses(G_in, G_in, 'w : 1', on_pre = on_pre ,method='linear', name='synapses_I')

S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

S_I_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre, method='linear')

# -------network topology----------
S_inE.connect(p = 0.3)
S_inI.connect(p = 0.3)
S_EE.connect(p = 0.2)
S_EI.connect(p = 0.2)
S_IE.connect(p = 0.2)
S_II.connect(p = 0.2)
S_I_readout.connect(j='i')
S_E_readout.connect(j='i')

G_ex.v = '13.5+1.5*rand()'
G_in.v = '13.5+1.5*rand()'

S_inE.w = 'i_inE*randn()+i_inE'
S_inI.w = 'i_inI*randn()+i_inI'
S_EE.w = 'i_EE*randn()+i_EE'
S_IE.w = 'i_IE*randn()+i_IE'
S_EI.w = 'i_EI*randn()+i_EI'
S_II.w = 'i_II*randn()+i_II'

S_EE.delay = 1.5
S_EI.delay = 0.8
S_IE.delay = 0.8
S_II.delay = 0.8

# ------monitor----------------
m_g_ex = StateMonitor(G_ex, (['I', 'v']), record=True)
m_g_in = StateMonitor(G_in, (['I', 'v']), record=True)
m_read = StateMonitor(G_readout, ('I'), record=True)
m_in = StateMonitor(Input, ('I'), record=True)

# ------create network-------------
net = Network(collect())
net.store('first')
fig_init_w =plt.figure(figsize=(4,16))
subplot(411)
brian_plot(S_EE.w)
subplot(421)
brian_plot(S_EI.w)
subplot(431)
brian_plot(S_IE.w)
subplot(441)
brian_plot(S_II.w)








