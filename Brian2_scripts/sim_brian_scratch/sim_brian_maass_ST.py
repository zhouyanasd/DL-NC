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
import pickle
from bqplot import *
import ipywidgets as widgets


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
        sum = np.sum(input[:, i * interval_: (i + 1) * interval_][:,::-step], axis=1)
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
            trans_cos = np.round(0.5*A*(np.cos(x+np.pi*(i/n))+1)).clip(0,A-1)
            coding = [([0] * trans_cos.shape[1]) for i in range(A*trans_cos.shape[0])]
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
    for value in encoding_latency_MNIST(data_frame_obj['value'][:n], kwargs['coding_n'], int(0.1*duration)):
        for data in value:
            data_frame_s.append(list(data))
        interval = duration - value.shape[0]
        if interval > 0.3*duration:
            data_frame_s.extend([[0] * 784] * interval)
        else:
            raise Exception('duration is too short')
    data_frame_s = np.asarray(data_frame_s)
    label = data_frame_obj['label'][:n]
    return data_frame_s, label


def result_save(path, *arg, **kwarg):
    fw = open(path, 'wb')
    pickle.dump(kwarg, fw)
    fw.close()


def result_pick(path):
    fr = open(path, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data


def animation(t, v, interval, duration, a_step=10, a_interval=100, a_duration = 10):

    xs = LinearScale()
    ys = LinearScale()

    line = Lines(x=t[:interval], y=v[:,:interval], scales={'x': xs, 'y': ys})
    xax = Axis(scale=xs, label='x', grid_lines='solid')
    yax = Axis(scale=ys, orientation='vertical', tick_format='0.2f', label='y', grid_lines='solid')

    fig = Figure(marks=[line], axes=[xax, yax], animation_duration=a_duration)

    def on_value_change(change):
        line.x = t[change['new']:interval+change['new']]
        line.y = v[:,change['new']:interval+change['new']]

    play = widgets.Play(
        interval=a_interval,
        value=0,
        min=0,
        max=duration,
        step=a_step,
        description="Press play",
        disabled=False
    )
    slider = widgets.IntSlider(min=0, max=duration)
    widgets.jslink((play, 'value'), (slider, 'value'))
    slider.observe(on_value_change, names='value')
    return play, slider, fig


def allocate(G, X, Y, Z):
    V = np.zeros((X,Y,Z), [('x',float),('y',float),('z',float)])
    V['x'], V['y'], V['z'] = np.meshgrid(np.linspace(0,X-1,X),np.linspace(0,X-1,X),np.linspace(0,Z-1,Z))
    V = V.reshape(X*Y*Z)
    np.random.shuffle(V)
    n = 0
    for g in G:
        for i in range(g.N):
            g.x[i], g.y[i], g.z[i]= V[n][0], V[n][1], V[n][2]
            n +=1
    return G


def w_norm2(n_post, Synapsis):
    for i in range(n_post):
        a = Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]]
        Synapsis.w[np.where(Synapsis._synaptic_post == i)[0]] = a/np.linalg.norm(a)



