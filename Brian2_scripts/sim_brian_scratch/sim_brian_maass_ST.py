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


class ST_classification_mass():
    def __init__(self, n_p, n, duration, frequency, dt):
        self.n_p = n_p
        self.n = n
        self.duration = duration
        self.frequency = frequency
        self.dt = dt
        self.n_1 = int(ceil(n * duration * frequency))
        self.n_0 = int(ceil(n * duration / dt)) - self.n_1
        self.D = int(duration / dt)
        self.pattern_generation()

    def pattern_generation(self):
        self.pattern = []
        for i in range(self.n_p):
            b = [1] * self.n_1
            b.extend([0] * self.n_0)
            c = np.array(b)
            np.random.shuffle(c)
            self.pattern.append(c)

    def data_generation(self):
        value = []
        label = []
        for i in range(self.n):
            select = np.random.choice(self.n_p)
            fragment = np.random.choice(self.n)
            value.extend(self.pattern[select][self.D * fragment:self.D * (fragment + 1)])
            label.append(select)
            data = {'label': label, 'value': value}
        return data

    def data_noise_jitter(self, data):
        pass

    def data_generation_batch(self, number, noise = False):
        value = []
        label = []
        for i in range(number):
            data = self.data_generation()
            if noise :
                self.data_noise_jitter(data)
            value.append(data['value'])
            label.append(data['label'])
            df = pd.DataFrame({'value': pd.Series(value), 'label': pd.Series(label)})
        return df

    def get_series_data(self, data_frame, is_order=True, *args, **kwargs):
        if not is_order:
            data_frame_obj = data_frame.sample(frac=1).reset_index(drop=True)
        else:
            data_frame_obj = data_frame
        data_frame_s = []
        for value in data_frame_obj['value']:
            data_frame_s.extend(value)
        data_frame_s = np.asarray(data_frame_s)
        label = data_frame_obj['label']
        return data_frame_s, list(map(list, zip(*label)))


# -----parameter setting-------
duration = 1000
N_train = 1000
N_test = 500
Dt = defaultclock.dt = 1*ms
sample = 1
pre_train_loop = 0

n = 108
R = 2
A_EE = 30
A_EI = 60
A_IE = -19
A_II = -19
A_inE = 18
A_inI = 9

ST = ST_classification_mass(2, 4, duration, 20*Hz, Dt)
df_train = ST.data_generation_batch(N_train)
df_test = ST.data_generation_batch(N_test)

data_train_s, label_train = ST.get_series_data(df_train, False)
data_test_s, label_test = ST.get_series_data(df_test, False)

duration_train = len(data_train_s) * Dt
duration_test = len(data_test_s) * Dt

equ_in = '''
I = stimulus(t,i) : 1
'''

equ = '''
dv/dt = (I-v) / (30*ms) : 1 (unless refractory)
dg/dt = (-g)/(3*ms) : 1
dh/dt = (-h)/(6*ms) : 1
I = (g+h)+13.5: 1
x : 1
y : 1
z : 1
'''

equ_read = '''
dv/dt = (I-v) / (30*ms) : 1
dg/dt = (-g)/(3*ms) : 1 
dh/dt = (-h)/(6*ms) : 1
I = (g+h): 1
'''




